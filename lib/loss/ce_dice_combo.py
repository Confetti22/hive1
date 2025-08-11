import torch
import torch.nn as nn
import torch.nn.functional as F



class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.alpha is not None:
            alpha = self.alpha.to(targets.device)
            at = alpha.gather(0, targets)
            loss *= at

        return loss.mean() if self.reduction == 'mean' else loss.sum()


class BinaryFocalLoss(nn.Module):
    """
    Focal loss for *single-logit* binary classification.
        inputs  : [N]   (raw scores)
        targets : [N]   (0 / 1)
    """
    def __init__(self, alpha: float | None = None, gamma: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha           # scalar weight for the positive class
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(inputs)                    # [N]
        # Prob. of the ground-truth class for each sample
        pt = torch.where(targets == 1, p, 1 - p)

        # Standard BCE (no reduction yet)
        bce = F.binary_cross_entropy_with_logits(inputs, targets.float(),
                                                 reduction='none')

        loss = (1 - pt) ** self.gamma * bce          # focal term
        if self.alpha is not None:
            loss = self.alpha * loss                 # scale positives

        return loss.mean() if self.reduction == 'mean' else loss.sum()


class ComboLoss(nn.Module):
    """
    Cross-Entropy/BCE + Dice.
    Works for both *multi-class* ([N, K]) and *binary* ([N]) logits.

    Parameters
    ----------
    binary         : set to True when using a single-logit binary setup
    focal          : switch CE/BCE → focal
    class_weights  :  ▸ multi-class   → Tensor[K]  (α_k for CE / softmax focal)
                      ▸ binary        → float      (pos_weight for BCE,
                                                    or α for focal)
    """
    def __init__(self,
                 weight_ce: float = 1.0,
                 weight_dice: float = 1.0,
                 smooth: float = 1e-6,
                 class_weights: torch.Tensor | float | None = None,
                 focal: bool = False,
                 binary: bool = False):
        super().__init__()
        self.weight_ce   = weight_ce
        self.weight_dice = weight_dice
        self.smooth      = smooth
        self.binary      = binary

        if binary:
            if focal:
                self.ce = BinaryFocalLoss(alpha=class_weights)  # scalar α
            else:
                # BCEWithLogitsLoss expects targets∈{0,1}
                self.ce = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            if focal:
                self.ce = FocalLoss(alpha=class_weights)        # original
            else:
                self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits_flat: torch.Tensor,
                      targets_flat: torch.Tensor) -> torch.Tensor:

        # ----- Cross-Entropy / BCE / Focal ----------------------------------
        ce_loss = self.ce(logits_flat, targets_flat if not self.binary
                          else targets_flat.float())

        # ----- Dice ---------------------------------------------------------
        if self.binary:                                          # [N]
            probs   = torch.sigmoid(logits_flat)                 # [N]
            targets = targets_flat.float()                       # [N]

            intersection = (probs * targets).sum()
            union        = probs.sum() + targets.sum()
            dice_loss    = 1 - (2 * intersection + self.smooth) / (union + self.smooth)

        else:                                                    # [N, K]
            probs     = torch.softmax(logits_flat, dim=1)        # [N, K]
            num_cls   = probs.shape[1]
            one_hot   = torch.zeros_like(probs).scatter_(1, targets_flat.unsqueeze(1), 1)

            intersection = (probs * one_hot).sum(dim=0)          # [K]
            union        = probs.sum(dim=0) + one_hot.sum(dim=0)
            dice_loss    = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
            dice_loss    = dice_loss.mean()

        # ----- Combo --------------------------------------------------------
        return self.weight_ce * ce_loss + self.weight_dice * dice_loss
