from __future__ import print_function, division
from typing import  List,Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(activation: str = 'relu') -> nn.Module:
    """Get the specified activation layer. 

    Args:
        activation (str): one of ``'relu'``, ``'leaky_relu'``, ``'elu'``, ``'gelu'``, 
            ``'swish'``, 'efficient_swish'`` and ``'none'``. Default: ``'relu'``
    """
    assert activation in ["relu", "leaky_relu", "elu", "gelu",
                          "swish", "efficient_swish", "none"], \
        "Get unknown activation key {}".format(activation)
    activation_dict = {
        "relu": nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.2, inplace=True),
        "elu": nn.ELU(alpha=1.0, inplace=True),
        "gelu": nn.GELU(),
        "none": nn.Identity(),
    }
    return activation_dict[activation]



def get_norm_3d(norm: str, out_channels: int, bn_momentum: float = 0.1) -> nn.Module:
    """Get the specified normalization layer for a 3D model.

    Args:
        norm (str): one of ``'bn'``, ``'sync_bn'`` ``'in'``, ``'gn'`` or ``'none'``.
        out_channels (int): channel number.
        bn_momentum (float): the momentum of normalization layers.
    Returns:
        nn.Module: the normalization layer
    """
    assert norm in ["bn", "sync_bn", "gn", "in", "none"], \
        "Get unknown normalization layer key {}".format(norm)
    if norm == "gn": assert out_channels%8 == 0, "GN requires channels to separable into 8 groups"
    norm = {
        "bn": nn.BatchNorm3d,
        "sync_bn": nn.SyncBatchNorm,
        "in": nn.InstanceNorm3d,
        "gn": lambda channels: nn.GroupNorm(8, channels),
        "none": nn.Identity,
    }[norm]
    if norm in ["bn", "sync_bn", "in"]:
        return norm(out_channels, momentum=bn_momentum)
    else:
        return norm(out_channels)


def conv3d_norm_act(in_planes, planes, kernel_size=(3, 3, 3), stride=1, groups=1,
                    dilation=(1, 1, 1), padding=(1, 1, 1), bias=True, pad_mode='replicate',
                    norm_mode='bn', act_mode='relu', trans= False,return_list=False):

    layers = []
    if trans:
        layers += [nn.ConvTranspose3d(in_planes, planes, kernel_size=kernel_size, stride=stride,
                         groups=groups, padding=padding, output_padding=1,padding_mode='zeros',
                         dilation=dilation, bias=bias)]
    else:
        layers += [nn.Conv3d(in_planes, planes, kernel_size=kernel_size, stride=stride,
                         groups=groups, padding=padding, padding_mode=pad_mode,
                         dilation=dilation, bias=bias)]

    layers += [get_norm_3d(norm_mode, planes)]
    layers += [get_activation(act_mode)]

    if return_list:  # return a list of layers
        return layers

    return nn.Sequential(*layers)

def transconv3d_norm_act(in_planes, planes, kernel_size=(3, 3, 3), stride=1, groups=1,
                    dilation=(1, 1, 1), padding=(0,0,0),output_padding=(1,1,1),
                    norm_mode='none', act_mode='elu', return_list=False):

    layers = []
    layers += [nn.ConvTranspose3d(in_planes, planes, kernel_size=kernel_size, stride=stride,
                         groups=groups, padding=padding,output_padding=output_padding, 
                         dilation=dilation)]
    layers += [get_norm_3d(norm_mode, planes)]
    layers += [get_activation(act_mode)]

    if return_list:  # return a list of layers
        return layers

    return nn.Sequential(*layers)


def norm_act_conv3d(in_planes, planes, kernel_size=(3, 3, 3), stride=1, groups=1,
                    dilation=(1, 1, 1), padding=(1, 1, 1), bias=False, pad_mode='replicate',
                    norm_mode='bn', act_mode='relu', return_list=False):

    layers = []
    layers += [get_norm_3d(norm_mode, in_planes)]
    layers += [get_activation(act_mode)]
    layers += [nn.Conv3d(in_planes, planes, kernel_size=kernel_size, stride=stride,
                         groups=groups, padding=padding, padding_mode=pad_mode,
                         dilation=dilation, bias=bias)]

    if return_list:  # return a list of layers
        return layers

    return nn.Sequential(*layers)


class SingleConv3d(nn.Module):
    def __init__(self,
                 in_planes: int,
                 planes: int,
                 stride: Union[int, tuple] = 1,
                 kernel_size : int =3 ,
                 padding : int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn',
                 ):
        super(SingleConv3d, self).__init__()
        self.conv = conv3d_norm_act(in_planes, planes, kernel_size=kernel_size, dilation=dilation,
                            stride=stride, groups=groups, padding=padding,
                            pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)

    def forward(self, x):
        y = self.conv(x)
        return y


class DoubleConv3d(nn.Module):
    def __init__(self,
                 in_planes: int,
                 planes: int,
                 kernel_size : int = 3,
                 padding : int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn',
                 ):
        super(DoubleConv3d, self).__init__()
        self.conv = nn.Sequential(
            conv3d_norm_act(planes, planes, kernel_size=kernel_size, dilation=dilation,
                            stride= 1, groups=groups, padding=padding,
                            pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            conv3d_norm_act(planes, planes, kernel_size=kernel_size, dilation=dilation,
                            stride= 1, groups=groups, padding=padding,
                            pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)
        )

    def forward(self, x):
        y = self.conv(x)
        return y

# ---------------------------
# Partial Convolution Layers
# ---------------------------
# Adapted from https://github.com/NVIDIA/partialconv/blob/master/models/partialconv3d.py


class PartialConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv3d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(
                self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2])
        else:
            self.weight_maskUpdater = torch.ones(
                1, 1, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
            self.weight_maskUpdater.shape[3] * self.weight_maskUpdater.shape[4]

        self.last_size = (None, None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 5
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(
                            input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3], input.data.shape[4]).to(input)
                    else:
                        mask = torch.ones(
                            1, 1, input.data.shape[2], input.data.shape[3], input.data.shape[4]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv3d(mask, self.weight_maskUpdater, bias=None,
                                            stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        # if self.update_mask.type() != input.type() or self.mask_ratio.type() != input.type():
        #     self.update_mask.to(input)
        #     self.mask_ratio.to(input)

        raw_out = super(PartialConv3d, self).forward(
            torch.mul(input, mask_in) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1, 1)
            output = torch.mul(raw_out - bias_view,
                               self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output

# -----------------------------
# Depthwise Convolution Layers
# -----------------------------


def get_conv(conv_type='standard'):
    assert conv_type in ['standard', 'partial']
    if conv_type == 'partial':
        return PartialConv3d
    return nn.Conv3d


def get_dilated_dw_convs(
    channels: int = 64,
    dilation_factors: List[int] = [1, 2, 4, 8],
    kernel_size: int = 3,
    stride: int = 1,
    conv_type: str = 'standard',
    pad_mode: str = 'zeros',
    isotropic: bool = False,
):
    assert channels % len(dilation_factors) == 0
    num_split = len(dilation_factors)
    conv_layer = dwconvkxkxk if isotropic else dwconv1xkxk
    return nn.ModuleList([
        conv_layer(
            channels // num_split,
            kernel_size,
            stride,
            conv_type=conv_type,
            padding_mode=pad_mode,
            dilation=dilation_factors[i])
        for i in range(num_split)
    ])


def dwconv1xkxk(planes, kernel_size=3, stride=1,
                dilation=1, conv_type='standard',
                padding_mode='zeros'):
    """1xkxk depthwise convolution with padding"""
    padding = ((kernel_size - 1) * dilation )// 2
    dilation = (1, dilation, dilation)
    padding = (0, padding, padding)
    stride = (1, stride, stride) if isinstance(stride, int) else stride
    return get_conv(conv_type)(
        planes,
        planes,
        kernel_size=(1, kernel_size, kernel_size),
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        groups=planes,
        bias=False,
        dilation=dilation)


def dwconvkxkxk(planes, kernel_size=3, stride=1,
                dilation=1, conv_type='standard',
                padding_mode='zeros'):
    """kxkxk depthwise convolution with padding"""
    padding = ((kernel_size - 1) * dilation )// 2
    return get_conv(conv_type)(
        planes,
        planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        groups=planes,
        bias=False,
        dilation=dilation)
