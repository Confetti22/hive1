import numpy as np
import torch

def compute_class_weights_from_dataset(dataset, num_classes, bnd=False, recon_target_flag=False):
    """
    update: 25/08/18 compatible to new_seg_dataset with 4 possible ouput
    Returns:
    - torch.Tensor of class weights.
    """
    class_counts = np.zeros(num_classes)

    for data in dataset:
        # Unpack according to the enabled flags
        if bnd and recon_target_flag:
            roi, mask, bnd_mask, recon_target = data
            label_vol = bnd_mask
        elif bnd and not recon_target_flag:
            roi, mask, bnd_mask = data
            label_vol = bnd_mask
        elif not bnd and recon_target_flag:
            roi, mask, recon_target = data
            label_vol = mask
        else:
            roi, mask = data
            label_vol = mask
        
        flat = label_vol[mask>= 0].flatten()
        counts = np.bincount(flat, minlength=num_classes)
        class_counts[:len(counts)] += counts

    total = class_counts.sum()
    class_weights = total / (num_classes * class_counts + 1e-6)  # Avoid division by zero
    return torch.tensor(class_weights, dtype=torch.float)