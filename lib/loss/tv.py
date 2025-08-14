import torch
import torch.nn as nn

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1.0):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        # Input shape: (B, C, H, W) or (B, C, D, H, W)
        # avoid D==1
        batch_size = x.size(0)
        spatial_dims = x.dim() - 2  # Number of spatial dims (2 for 2D, 3 for 3D)
        tv = 0.0

        if spatial_dims == 2:
            # (B, C, H, W)
            h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
            w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
            count_h = self._tensor_size(x[:, :, 1:, :])
            count_w = self._tensor_size(x[:, :, :, 1:])
            tv = (h_tv / count_h + w_tv / count_w)
        elif spatial_dims == 3:
            # (B, C, D, H, W)
            d_tv = torch.pow(x[:, :, 1:, :, :] - x[:, :, :-1, :, :], 2).sum()
            h_tv = torch.pow(x[:, :, :, 1:, :] - x[:, :, :, :-1, :], 2).sum()
            w_tv = torch.pow(x[:, :, :, :, 1:] - x[:, :, :, :, :-1], 2).sum()
            count_d = self._tensor_size(x[:, :, 1:, :, :])
            count_h = self._tensor_size(x[:, :, :, 1:, :])
            count_w = self._tensor_size(x[:, :, :, :, 1:])
            tv = (d_tv / count_d + h_tv / count_h + w_tv / count_w)
        else:
            raise ValueError("Unsupported input dimensions. Expected 4D or 5D input.")

        return self.TVLoss_weight * 2 * tv / batch_size

    def _tensor_size(self, t):
        return t.numel()