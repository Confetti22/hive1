# %%
#!/usr/bin/env python3
"""
Merge 8 per-layer binary masks (0/1) into a single multi-class label TIFF per slice.

Directory layout (example):
base_dir/
    L1/
        L1-Z55200.png
        L1-Z55500.png
        ...
    L23/
        L23-Z55200.png
        ...
    ...
    L6/
        L6-Z69600.png

Output:
base_dir/merged_bnds/
    merged-Z55200.tif
    merged-Z55500.tif
    ...
(Optionally) merged_stack.tif  (3-D volume, Z-axis ordered ascending)

Classes:
0 = background
1..8 = L1, L23, L4A, L4B, L4Ca, L4Cb, L5, L6
"""

from pathlib import Path
from skimage import io
import numpy as np

LAYER_ORDER = ["L1", "L23", "L4A", "L4B", "L4Ca", "L4Cb", "L5", "L6"]

def merge_layer_masks(base_dir: str,
                      z_start: int = 55200,
                      z_stop: int = 69600,
                      z_step: int = 300,
                      out_dir_name: str = "merged_bnds",
                      make_stack: bool = True,
                      overwrite: bool = False):
    base = Path(base_dir)
    out_dir = base / out_dir_name
    out_dir.mkdir(exist_ok=True)

    z_indices = list(range(z_start, z_stop + 1, z_step))
    stack_slices = []

    for layer in LAYER_ORDER:
        layer_dir = base / layer
        if not layer_dir.is_dir():
            raise FileNotFoundError(f"Missing directory: {layer_dir}")

    for zi in z_indices:
        out_path = out_dir / f"merged-Z{zi}.tif"
        if out_path.exists() and not overwrite:
            merged = io.imread(out_path)
            stack_slices.append(merged)
            print(f"[SKIP] {out_path.name} exists.")
            continue

        layer_arrays = []
        shape_ref = None

        for class_id, layer in enumerate(LAYER_ORDER, start=1):
            fpath = base / layer / f"{layer}-Z{zi}.png"
            if not fpath.is_file():
                raise FileNotFoundError(f"Expected file missing: {fpath}")

            arr = io.imread(fpath)

            if arr.ndim == 3:
                if arr.shape[2] == 1:
                    arr = arr[..., 0]
                else:
                    raise ValueError(f"{fpath} has {arr.shape[2]} channels; expected single-channel 0/255 mask.")

            if shape_ref is None:
                shape_ref = arr.shape
            elif arr.shape != shape_ref:
                raise ValueError(f"Shape mismatch at {fpath}: {arr.shape} vs {shape_ref}")

            uniq = np.unique(arr)
            if not np.all(np.isin(uniq, [0, 1, 255])):
                raise ValueError(f"{fpath} contains unexpected values {uniq}; expected {{0,255}} or already 0/1.")

            arr = (arr == 255).astype(np.uint8)  # normalize

            layer_arrays.append((class_id, arr))

        merged = np.zeros(shape_ref, dtype=np.uint8)

        for class_id, bin_mask in layer_arrays:
            overlap = (merged != 0) & (bin_mask == 1)
            if np.any(overlap):
                raise ValueError(
                    f"Overlap detected adding class {class_id} ({LAYER_ORDER[class_id-1]}) at Z{zi}; "
                    f"{overlap.sum()} pixels overlap."
                )
            merged[bin_mask == 1] = class_id

        io.imsave(out_path, merged, check_contrast=False)
        stack_slices.append(merged)
        print(f"[OK] Wrote {out_path.name} (shape={merged.shape})")

    if make_stack:
        stack = np.stack(stack_slices, axis=0)
        io.imsave(out_dir / "merged_stack.tif", stack.astype(np.uint8), check_contrast=False)
        print(f"[OK] Wrote merged_stack.tif shape={stack.shape}")

    with (out_dir / "label_mapping.txt").open("w") as f:
        f.write("Label mapping (uint8):\n0\tBackground\n")
        for i, layer in enumerate(LAYER_ORDER, start=1):
            f.write(f"{i}\t{layer}\n")

if __name__ == "__main__":
    # Adjust base_dir if different
    base_dir = "/home/confetti/data/rm009/rm009_roi/single_layer/merged_bnds/upscaled_before_smooth"
    merge_layer_masks(base_dir,
                      z_start=55200,
                      z_stop=69600,
                      z_step=300,
                      out_dir_name="merged_bnds",
                      make_stack=False,
                      overwrite=False)

