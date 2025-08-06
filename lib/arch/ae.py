import torch.nn as nn
import torch



def get_norm_nd(norm: str, out_channels: int, dim: int = 3, bn_momentum: float = 0.1) -> nn.Module:
    """Return a normalization layer for N-D convolution.

    Args:
        norm (str): One of ['bn', 'sync_bn', 'in', 'gn', 'none'].
        out_channels (int): Number of output channels.
        dim (int): Dimension of data (1, 2, or 3).
        bn_momentum (float): Momentum for BatchNorm or SyncBatchNorm.

    Returns:
        nn.Module: Normalization layer.
    """
    norm = norm.lower()
    assert norm in ["bn", "sync_bn", "in", "gn", "none"], f"Unknown normalization type: {norm}"
    assert dim in [1, 2, 3], f"Unsupported dimension: {dim}"

    norm_layers = {
        "bn": [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d],
        "sync_bn": [nn.SyncBatchNorm] * 3,
        "in": [nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d],
    }

    if norm in norm_layers:
        NormLayer = norm_layers[norm][dim - 1]
        return NormLayer(out_channels, momentum=bn_momentum)

    if norm == "gn":
        assert out_channels % 8 == 0, "GroupNorm requires out_channels divisible by 8"
        return nn.GroupNorm(8, out_channels)

    return nn.Identity()




def get_activation(act: str = 'relu') -> nn.Module:
    """Return an activation layer.

    Args:
        act (str): One of ['relu', 'leaky_relu', 'elu', 'gelu', 'swish', 'efficient_swish', 'none'].

    Returns:
        nn.Module: Activation layer.
    """
    act = act.lower()
    assert act in ["relu", "leaky_relu", "elu", "gelu", "swish", "efficient_swish", "none"], \
        f"Unknown activation type: {act}"

    activation_dict = {
        "relu": nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(0.2, inplace=True),
        "elu": nn.ELU(inplace=True),
        "gelu": nn.GELU(),
        "swish": nn.SiLU(),  # swish = SiLU
        "efficient_swish": nn.SiLU(),
        "none": nn.Identity()
    }

    return activation_dict[act]


def conv_nd_norm_act(in_channels, out_channels,
                     kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True,
                     dim=3, trans=False,
                     pad_mode='replicate', norm_mode='bn', act_mode='relu',
                     return_list=False):

    assert dim in [1, 2, 3], "Only 1D, 2D, or 3D convolutions are supported"

    # Padding mode compatibility
    if pad_mode not in ['zeros', 'reflect', 'replicate', 'circular']:
        pad_mode = 'zeros'


    # Dynamically pick Conv or ConvTranspose
    if trans:
        Conv = [nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d][dim - 1]
        conv_layer = Conv(
        in_channels, out_channels,
        kernel_size=kernel_size, stride=stride,
        padding=padding, dilation=dilation,
        groups=groups, bias=bias,
        padding_mode='zeros',  # transposed conv only supports 'zeros'
        output_padding=1,
    )
    else:
        Conv = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dim - 1]
        conv_layer = Conv(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation,
            groups=groups, bias=bias,
            padding_mode=pad_mode, 
        )

    norm_layer = get_norm_nd(norm_mode, out_channels, dim)
    act_layer = get_activation(act_mode)

    layers = [conv_layer, norm_layer, act_layer]

    return layers if return_list else nn.Sequential(*layers)


def make_block(in_ch, out_ch, ks, stride, padding, block_type, dim, trans, shared_kwargs):
    def conv(in_c, out_c):
        return conv_nd_norm_act(
            in_c, out_c, ks,
            stride if in_c == in_ch else 1,
            padding,
            dim=dim,
            trans=trans if in_c == in_ch else False,
            **shared_kwargs
        )

    if block_type == 'double':
        return nn.Sequential(conv(in_ch, out_ch), conv(out_ch, out_ch))
    elif block_type == 'triple':
        return nn.Sequential(conv(in_ch, out_ch), conv(out_ch, out_ch), conv(out_ch, out_ch))
    else:  # 'single'
        return nn.Sequential(conv(in_ch, out_ch))

# --------------------------------------------
#  Base AutoEncoder Class (ND)
# --------------------------------------------

class EncoderND(nn.Module):
    def __init__(self, in_channel, filters, kernel_size, dimension=3,
                 pad_mode='reflect', act_mode='elu', norm_mode='gn', block_type='double',avg_pool_size = None, avg_pool_padding=False,last_encoder=True):
        super().__init__()
        self.dim = dimension
        self.depth = len(filters)
        self.avg_pool_size = avg_pool_size
        self.avg_pool_padding = avg_pool_padding

        Conv = nn.Conv3d if dimension == 3 else nn.Conv2d

        self.shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode
        }

        k = kernel_size[0]
        p = int((k - 1) // 2)
        self.conv_in = conv_nd_norm_act(in_channel, filters[0], k, 2, p, dim=dimension, **self.shared_kwargs)

        self.down_layers = nn.ModuleList()
        for i in range(self.depth - 1):
            ks = kernel_size[min(i + 1, len(kernel_size) - 1)]
            p = int((ks - 1) // 2)
            block = make_block(filters[i], filters[i + 1], ks, 2, p, block_type, dim=dimension, trans=False,
                               shared_kwargs=self.shared_kwargs)
            self.down_layers.append(block)

        if last_encoder:
            self.last_encoder_conv = Conv(filters[-1], filters[-1], kernel_size=1)
        else:
            self.last_encoder_conv = None

    def forward(self, x):
        x = self.conv_in(x)
        for layer in self.down_layers:
            x = layer(x)
        if self.last_encoder_conv:
            x = self.last_encoder_conv(x)

        avg_pool_size = self.avg_pool_size
        apply_avg_flag = self.avg_pool_size if (self.avg_pool_size is None) else avg_pool_size[0]
        if apply_avg_flag:
            
            if self.avg_pool_padding:
                pad = [int((x - 1)//2) for x in avg_pool_size]
            else:
                pad =0
            pool = nn.AvgPool3d(kernel_size=avg_pool_size, stride=1,padding=pad) if self.dim == 3 else nn.AvgPool2d(kernel_size=avg_pool_size, stride=1,padding=pad)
            x = pool(x)
        return x

class DecoderND(nn.Module):
    def __init__(self, out_channel, filters, kernel_size, dimension=3,
                 pad_mode='reflect', act_mode='elu', norm_mode='gn', block_type='double'):
        super().__init__()
        self.dim = dimension
        self.depth = len(filters)

        ConvTrans = nn.ConvTranspose3d if dimension == 3 else nn.ConvTranspose2d

        self.shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode
        }

        self.up_layers = nn.ModuleList()
        for i in reversed(range(self.depth - 1)):
            ks = kernel_size[i + 1]
            p = int((ks - 1) // 2)
            block = make_block(filters[i + 1], filters[i], ks, stride=2, padding=p,
                               block_type=block_type, dim=dimension, trans=True,
                               shared_kwargs=self.shared_kwargs)
            self.up_layers.append(block)

        k = kernel_size[0]
        p = int((k - 1) // 2)
        self.conv_out = nn.Sequential(
            ConvTrans(filters[0], out_channel, kernel_size=k, stride=2, padding=p, output_padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        for layer in self.up_layers:
            x = layer(x)
        x = self.conv_out(x)
        return x
    

class BaseAutoEncoderND(nn.Module):
    def __init__(self, in_channel, out_channel, filters, kernel_size, dimension=3,
                 pad_mode='reflect', act_mode='elu', norm_mode='gn', block_type='double'):
        super().__init__()
        self.encoder = EncoderND(in_channel, filters, kernel_size, dimension,
                                 pad_mode, act_mode, norm_mode, block_type)
        self.decoder = DecoderND(out_channel, filters, kernel_size, dimension,
                                 pad_mode, act_mode, norm_mode, block_type)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AutoEncoder3D(BaseAutoEncoderND):
    def __init__(self, **kwargs):
        super().__init__(dimension=3, **kwargs)


class AutoEncoder2D(BaseAutoEncoderND):
    def __init__(self, **kwargs):
        super().__init__(dimension=2, **kwargs)

# --------------------------------------------
# MLP 
# --------------------------------------------


class MLP(nn.Module):
    def __init__(self, filters=[24, 18, 12, 8]):
        super(MLP, self).__init__()
        
        layers = []
        for in_features, out_features in zip(filters[:-1], filters[1:]):
            layers.append(nn.Linear(in_features, out_features))
        
        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        x = self.layers[-1](x)  # Last layer, no activation
        return x / x.norm(p=2, dim=-1, keepdim=True)


## 1*1 conv version of the above MLP

class ConvMLP(nn.Module):
    def __init__(self, filters=[24, 18, 12, 8], dims=2):
        super(ConvMLP, self).__init__()

        assert dims in [2, 3], "dims must be 2 or 3"
        Conv = nn.Conv2d if dims == 2 else nn.Conv3d

        layers = []
        for in_channels, out_channels in zip(filters[:-1], filters[1:]):
            layers.append(Conv(in_channels, out_channels, kernel_size=1))

        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dims = dims  # Save dims for later use (e.g., normalization)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        x = self.layers[-1](x)  # Last layer, no activation

        # L2 normalization across channel dim
        x = x / x.norm(p=2, dim=1, keepdim=True)
        return x

class ComposedModel(nn.Module):
    def __init__(self, in_channel,cnn_filters, kernel_size,dims,mlp_filters, 
                 pad_mode='reflect', act_mode='elu', norm_mode='gn', block_type='double',avg_pool_size= None, avg_pool_padding=None,last_encoder=True):
        super().__init__()
        self.cnn_encoder = EncoderND(in_channel, cnn_filters, kernel_size, dims,
                                 pad_mode, act_mode, norm_mode, block_type,avg_pool_size=avg_pool_size,avg_pool_padding =avg_pool_padding,last_encoder=last_encoder)
        self.mlp_encoder = ConvMLP(mlp_filters,dims)

    def forward(self, x):
        x = self.cnn_encoder(x) # B*C*H*W --> B*H*W*C --> (B*H*W)*C
        # x = x.permute(0, 2, 3, 1)
        # x = x.reshape(-1, x.shape[-1])
        x = self.mlp_encoder(x)
        return x



MODEL_MAP = {
    'ae2': AutoEncoder2D,
    'ae3': AutoEncoder3D,
    'encoder':EncoderND,
}




def build_autoencoder_model(args):

    model_arch = args.model_name
    assert model_arch in MODEL_MAP.keys()
    kwargs = {
        'in_channel': args.in_planes,
        'out_channel': args.out_planes,
        'filters': args.filters,
        'kernel_size': args.kernel_size,
        'pad_mode': args.pad_mode,
        'act_mode': args.act_mode,
        'norm_mode': args.norm_mode,
        'block_type': args.block_type,
    }

    model = MODEL_MAP[args.model_name](**kwargs)
    print('model: ', model.__class__.__name__)

    return model


def build_encoder_model(args,dims):

    kwargs = {
        'in_channel': args.in_planes,
        'filters': args.filters,
        'kernel_size': args.kernel_size,
        'pad_mode': args.pad_mode,
        'act_mode': args.act_mode,
        'norm_mode': args.norm_mode,
        'block_type': args.block_type,
        'avg_pool_size':args.avg_pool_size,
        'last_encoder': args.last_encoder,
    }

    model = EncoderND(dimension=dims,**kwargs)

    return model 

def build_final_model(args):
    kwargs = {
        'in_channel': args.in_planes,
        'cnn_filters': args.filters,
        'kernel_size': args.kernel_size,
        'pad_mode': args.pad_mode,
        'act_mode': args.act_mode,
        'norm_mode': args.norm_mode,
        'block_type': args.block_type,
        'dims':args.dims,
        'mlp_filters':args.mlp_filters,
        'avg_pool_size':args.avg_pool_size,
        'avg_pool_padding':args.avg_pool_padding,
        'last_encoder': args.last_encoder,
    }
    model = ComposedModel(**kwargs)
    return model
    


def modify_key(weight_dict,source,target):
    new_weight_dict = {}
    for key, value in weight_dict.items():
        new_key = key.replace(source,target)
        new_weight_dict[new_key] = value
    return new_weight_dict


def delete_key(weight_dict,pattern_lst:tuple):
    new_weight_dict = {k: v for k, v in weight_dict.items() if not k.startswith(pattern_lst)}
    return new_weight_dict 

def load_encoder2encoder(model,ckpt_pth):
    ckpt = torch.load(ckpt_pth)
    removed_module_dict = modify_key(ckpt,source='module.',target='')
    load_result = model.load_state_dict(removed_module_dict, strict=False)

    missing = load_result.missing_keys
    unexpected = load_result.unexpected_keys

    if not missing and not unexpected:
        print("load_encoder2encoder:✅ All weights loaded successfully.")
    else:
        print("load_encoder2encoder:⚠️ Some weights were not loaded exactly:")
        if missing:
            print(f"   • Missing keys ({len(missing)}):\n     {missing}")
        if unexpected:
            print(f"   • Unexpected keys ({len(unexpected)}):\n     {unexpected}")


    return load_result

def load_ae2encoder(model,ckpt_pth):
    ckpt = torch.load(ckpt_pth)
    removed_module_dict = modify_key(ckpt['model'],source='module.encoder.',target='')
    load_result = model.load_state_dict(removed_module_dict, strict=False)

    missing = load_result.missing_keys
    unexpected = load_result.unexpected_keys

    if not missing and not unexpected:
        print("✅ All weights loaded successfully.")
    else:
        print("⚠️ Some weights were not loaded exactly:")
        if missing:
            print(f"   • Missing keys ({len(missing)}):\n     {missing}")
        if unexpected:
            print(f"   • Unexpected keys ({len(unexpected)}):\n     {unexpected}")

    return load_result


def load_mlpencoder_dict(model,ckpt_pth):
    ckpt = torch.load(ckpt_pth)
    #remove any 'module.' keywords if exist in weights_pth and remove unwanted layers
    load_result = model.load_state_dict(ckpt,strict=False)

    # 4. Inspect missing / unexpected keys
    missing = load_result.missing_keys
    unexpected = load_result.unexpected_keys

    if not missing and not unexpected:
        print("load_mlpencoder_dict ✅ All weights loaded successfully.")
    else:
        print("load_mlpencoder_dict ⚠️ Some weights were not loaded exactly:")
        if missing:
            print(f"   • Missing keys ({len(missing)}):\n     {missing}")
        if unexpected:
            print(f"   • Unexpected keys ({len(unexpected)}):\n     {unexpected}")

    return load_result


def load_mlp_ckpt_to_convmlp(convmlp_model, mlp_ckpt_pth=None, mlp_weight_dict=None, dims=2):
    if mlp_ckpt_pth is not None:
        mlp_ckpt = torch.load(mlp_ckpt_pth)
    elif mlp_weight_dict is not None:
        mlp_ckpt = mlp_weight_dict
    else:
        raise ValueError("Either 'mlp_ckpt_pth' or 'mlp_weight_dict' must be provided.")

    conv_state_dict = convmlp_model.state_dict()
    new_state_dict = {}

    linear_idx = 0
    for name, param in conv_state_dict.items():
        if 'weight' in name:
            linear_w = mlp_ckpt[f'layers.{linear_idx}.weight']  # shape: [out, in]
            if dims ==3:
                new_w = linear_w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # shape: [out, in, 1, 1]
            else:
                new_w = linear_w.unsqueeze(-1).unsqueeze(-1)  # shape: [out, in, 1, 1]
            new_state_dict[name] = new_w
        elif 'bias' in name:
            linear_b = mlp_ckpt[f'layers.{linear_idx}.bias']
            new_state_dict[name] = linear_b
            linear_idx += 1  # advance to next Linear layer
        else:
            raise ValueError(f'Unknown param name {name}')

    convmlp_model.load_state_dict(new_state_dict)
    print(f"load_mlp_ckpt_to_convmlp ✅all weiths loaded into convmlp successfully")

def load_compose_encoder_dict(cmodel,cnn_ckpt_pth,mlp_ckpt_pth=None,mlp_weight_dict=None,dims=2):
    cnn = cmodel.cnn_encoder
    mlp = cmodel.mlp_encoder
    load_encoder2encoder(cnn,cnn_ckpt_pth)
    # load_mlpencoder_dict(mlp,mlp_ckpt_pth)
    load_mlp_ckpt_to_convmlp(mlp,mlp_ckpt_pth,mlp_weight_dict,dims)
    







