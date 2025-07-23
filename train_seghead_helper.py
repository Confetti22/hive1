import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Union

def seg_valid(img_logger,valid_loader,seg_head,epoch,device,loss_fn):
    seg_head.eval()

    valid_loss  = []
    gt_maskes   = []   # list of 2-D numpy arrays
    pred_maskes = []
    total_top1 = []
    total_top3= []

    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader):          # inputs: [B,C,D,H,W], labels: [B,D,H,W]
            inputs  = inputs.to(device)
            labels  = labels.to(device)

            logits  = seg_head(inputs)                     # [B,K,D,H,W]
            mask    = labels >= 0                          # same shape as labels

            # ---------- loss ----------
            # bring class-channel last → choose only valid voxels
            logits_flat  = logits.permute(0, 2, 3, 4, 1)[mask]   # [N_vox, K]
            labels_flat  = labels[mask]
            loss         = loss_fn(logits_flat, labels_flat)
            valid_loss.append(loss.item())
        
            top1, top3 = accuracy(logits_flat, labels_flat, topk=(1, 3))
            total_top1.append(top1.item())
            total_top3.append(top3.item())

            # ---------- prediction ----------
            probs = F.softmax(logits, dim=1)               # softmax over channel K
            pred  = torch.argmax(probs, dim=1) + 1         # [B,D,H,W], +1 keeps 0 for ignore
            pred  = pred * mask                            # zero-out ignore voxels

            # ---------- move to CPU once ----------
            pred_np  = pred.cpu().numpy()                  # [B,D,H,W]
            label_np = (labels + 1).cpu().numpy()          # shift GT to match pred range

            z_mid = pred_np.shape[1] // 2                  # central axial slice index

            # one 2-D slice per sample
            gt_maskes.extend(label_np[:, z_mid, :, :])
            pred_maskes.extend(pred_np[:, z_mid, :, :])

    num_classes = 8                       # your label count
    cmap        = plt.get_cmap('nipy_spectral', num_classes)
    max_cols    = 4                       # ≤ 4 columns in the gallery

    # --- 1) merge GT + prediction for visualisation ----------------------------
    combined_imgs = []
    for gt, pred in zip(gt_maskes, pred_maskes):
        # Put GT on the left, prediction on the right
        combined = np.hstack((gt, pred))         # shape: (H, 2*W)
        combined_imgs.append(combined)

    # --- 2) build a grid -------------------------------------------------------
    n_imgs  = len(combined_imgs)
    n_cols  = min(max_cols, n_imgs)
    n_rows  = math.ceil(n_imgs / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,figsize=(4 * n_cols, 4 * n_rows),squeeze=False)

    for idx, img in enumerate(combined_imgs):
        r, c = divmod(idx, n_cols)
        axes[r, c].imshow(img, cmap=cmap, vmin=1, vmax=num_classes)
        axes[r, c].set_title(f"Sample {idx}", fontsize=10)
        axes[r, c].axis("off")

    # Hide any empty cells (when images % max_cols ≠ 0)
    for idx in range(n_imgs, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    fig.tight_layout()

    img_logger.add_figure('gt/pred',fig,global_step = epoch)

    avg_valid_loss = sum(valid_loss)/len(valid_loss)
    avg_top1 = sum(total_top1)/len(total_top1)
    avg_top3 = sum(total_top3)/len(total_top3)

    seg_head.train()
    return avg_valid_loss, avg_top1,avg_top3


import numpy as np
import torch

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

import torch
import torch.nn as nn
import torch.nn.functional as F


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



def accuracy(
    logits_flat: torch.Tensor,         # [N, K]
    targets_flat: torch.Tensor,        # [N]
    topk: Union[int, Sequence[int]] = 1,
) -> Union[float, list[float]]:
    """
    Compute (top-k) accuracy for already-flattened logits and integer labels.

    Parameters
    ----------
    logits_flat : Tensor, shape (N, K)
        Raw, un-normalized model outputs.  (Softmax not required.)
    targets_flat : Tensor, shape (N,)
        Ground-truth class indices  (0 ≤ label < K).
    topk : int | sequence<int>, default 1
        k or list/tuple of ks for top-k accuracy.

    Returns
    -------
    float | list[float]
        If `topk` is a single int: scalar accuracy in [0, 1].
        If `topk` is a sequence: list with accuracies for each k in order.
    """
    if isinstance(topk, int):
        topk = (topk,)

    with torch.no_grad():
        maxk = max(topk)
        # Shape: (N, maxk) → transpose to (maxk, N)
        _, pred = logits_flat.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()

        correct = pred.eq(targets_flat.unsqueeze(0).expand_as(pred))

        accs = []
        for k in topk:
            # Flatten first k rows, count correct, normalise by N
            correct_k = correct[:k].reshape(-1).float().sum()
            accs.append((correct_k / logits_flat.size(0)).item())

    return accs[0] if len(accs) == 1 else accs