import torch
import torch.nn as nn

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