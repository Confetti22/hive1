import torch
from typing import Sequence, Tuple, Union, Literal, List

def accuracy(
    logits_flat: torch.Tensor,         # [N, K]
    targets_flat: torch.Tensor,        # [N]
    topk: Union[int, Sequence[int]] = 1,
) -> Union[float, list[float]]:
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