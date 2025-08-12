import numpy as np
import torch
def compute_class_weights_from_dataset(dataset, num_classes,bnd=False):
    # Assume dataset returns (image, label), and label is a 3D/4D tensor
    class_counts = np.zeros(num_classes)

    if bnd:
        for roi, mask,bnd in dataset:
            # Flatten and count valid pixels only (label >= 0)
            flat = bnd[mask >= 0].flatten()
            counts = np.bincount(flat, minlength=num_classes)
            class_counts[:len(counts)] += counts
    else:
        for roi, mask in dataset:
            # Flatten and count valid pixels only (label >= 0)
            flat = mask[mask >= 0].flatten()
            counts = np.bincount(flat, minlength=num_classes)
            class_counts[:len(counts)] += counts

    total = class_counts.sum()
    class_weights = total / (num_classes * class_counts + 1e-6)  # inverse frequency
    return torch.tensor(class_weights, dtype=torch.float)