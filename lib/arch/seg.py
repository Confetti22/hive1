
import torch.nn as nn

class SegmentationHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 12),
            nn.ReLU(),
            nn.Linear(12, num_classes)  # Multiclass logits
        )

    def forward(self, x):
        return self.classifier(x)
    

class ConvSegHead(nn.Module):
    """
    A small 3D convolutional head for voxel-wise classification.
    adding padding to insure the output shape is the same as input
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv3d(in_channels // 2, num_classes, kernel_size=1, padding=0)
        )

    def forward(self, x):
        # x shape: [B, C, D, H, W]
        return self.net(x)  # logits per class
