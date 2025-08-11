import torch.nn as nn

import torch
import torch.nn.functional as F
import numpy as np

class Loss(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.loss_fn = nn.L1Loss(reduction = 'mean')

    def forward(self, preds, labels):
        loss = self.loss_fn(preds, labels)
        return loss

def get_loss(args):
    return Loss(args)



def compute_class_weights_from_dataset(dataset, num_classes):
    # Assume dataset returns (image, label), and label is a 3D/4D tensor
    class_counts = np.zeros(num_classes)

    for roi, mask,bnd in dataset:
        # Flatten and count valid pixels only (label >= 0)
        flat = bnd[mask >= 0].flatten()
        counts = np.bincount(flat, minlength=num_classes)
        class_counts[:len(counts)] += counts

    total = class_counts.sum()
    class_weights = total / (num_classes * class_counts + 1e-6)  # inverse frequency
    return torch.tensor(class_weights, dtype=torch.float)



class WeightedL1Loss(nn.Module):
    """
    Pixel-wise L1 loss with per-class weights.

    Parameters
    ----------
    class_weights : 1-D tensor, shape (C,)
        Weight for each label value 0 … C-1.
        For binary tasks, pass tensor([w_neg, w_pos]).
    reduction : {'mean', 'sum', 'none'}
    logits   : bool
        • True  – preds are raw logits → will apply sigmoid  
        • False – preds are already probabilities in [0,1]
    """
    def __init__(self,
                 class_weights: torch.Tensor,
                 reduction: str = "mean",
                 logits: bool = True):
        super().__init__()

        # keep weights on the module so they move with .to(device)
        self.register_buffer("class_weights", class_weights.float())
        self.reduction = reduction
        self.logits    = logits

    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.logits:
            preds = torch.sigmoid(preds)          # (N,1,H,W) → probs in [0,1]

        # build a weight map the same shape as labels
        w_pos = self.class_weights[1]
        w_neg = self.class_weights[0]
        weights = torch.where(labels > 0.5, w_pos, w_neg)

        loss = weights * torch.abs(preds - labels)  # weighted MAE

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:                     # 'none'
            return loss
