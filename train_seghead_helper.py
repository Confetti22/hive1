import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt

def seg_valid(img_logger,valid_loader,seg_head,epoch,device,loss_fn):
    seg_head.eval()

    valid_loss  = []
    gt_maskes   = []   # list of 2-D numpy arrays
    pred_maskes = []

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

    seg_head.train()
    return avg_valid_loss


import numpy as np
import torch

def compute_class_weights_from_dataset(dataset, num_classes):
    # Assume dataset returns (image, label), and label is a 3D/4D tensor
    class_counts = np.zeros(num_classes)

    for _, label in dataset:
        # Flatten and count valid pixels only (label >= 0)
        flat = label[label >= 0].flatten()
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

class ComboLoss(nn.Module):
    """
    CE + Dice loss that expects:
        logits_flat : [N, K]   (after any masking / flattening)
        targets_flat: [N]      (integer class labels)
    """
    def __init__(
        self,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
        smooth: float = 1e-6,
        class_weights: torch.Tensor | None = None,  # optional α for CE
        focal = False,
    ):
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.smooth = smooth
        if focal:
            self.ce = FocalLoss(alpha=class_weights)
        else:
            self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits_flat: torch.Tensor, targets_flat: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        logits_flat : [N, K]  raw scores
        targets_flat: [N]     long
        """
        # ---------- Cross-Entropy ----------
        ce_loss = self.ce(logits_flat, targets_flat)

        # ---------- Dice ----------
        probs = torch.softmax(logits_flat, dim=1)          # [N, K]
        num_classes = probs.shape[1]
        one_hot = torch.zeros_like(probs).scatter_(1, targets_flat.unsqueeze(1), 1)

        # per-class Dice, then mean
        intersection = (probs * one_hot).sum(dim=0)        # [K]
        union        = probs.sum(dim=0) + one_hot.sum(dim=0)
        dice_loss = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = dice_loss.mean()

        # ---------- Combo ----------
        return self.weight_ce * ce_loss + self.weight_dice * dice_loss