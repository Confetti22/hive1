import torch
from typing import Sequence, Tuple, Union, Literal, List

from typing import Sequence, Union
import torch

def accuracy(
    logits_flat: torch.Tensor,          # (N, K)  or  (N,) / (N,1)
    targets_flat: torch.Tensor,         # (N,)    int labels 0/1/…/K-1
    topk: Union[int, Sequence[int]] = 1,
    threshold: float = 0.5,             # only used when logits is 1-D
) -> Union[float, list[float]]:
    """
    Classification / segmentation accuracy.

    • If logits_flat has K >= 2 columns → top-k accuracy like ImageNet.
    • If K == 1 or logits_flat is 1-D  → binary accuracy with `sigmoid`.

    Returns
    -------
    float | list[float]   Same contract as the original function.
    """
    # -------- binary case ------------------------------------------------
    if logits_flat.ndim == 1 or logits_flat.shape[1] == 1:
        # squeeze to 1-D if needed
        if logits_flat.ndim == 2:
            logits_flat = logits_flat.squeeze(1)

        with torch.no_grad():
            probs  = torch.sigmoid(logits_flat)
            preds  = (probs >= threshold).long()
            correct = (preds == targets_flat).float().sum()
            acc = (correct / targets_flat.numel()).item()
        return acc if isinstance(topk, int) else [acc]

    # -------- multi-class case (K >= 2) ----------------------------------
    if isinstance(topk, int):
        topk = (topk,)

    with torch.no_grad():
        maxk = max(topk)
        # Shape: (N, maxk) → transpose to (maxk, N)
        _, pred = logits_flat.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()                                    # (maxk, N)

        correct = pred.eq(targets_flat.unsqueeze(0).expand_as(pred))

        accs = []
        for k in topk:
            # Flatten first k rows, count correct, normalise by N
            correct_k = correct[:k].reshape(-1).float().sum()
            accs.append((correct_k / logits_flat.size(0)).item())

    return accs[0] if len(accs) == 1 else accs