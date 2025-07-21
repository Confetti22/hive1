# import itk
# import numpy as np
# import tifffile as tif
# # image = itk.imread("/home/confetti/data/rm009/rm009_roi/all-z64800-65104/All-Z64800-65104.mha")
# # size = itk.size(image)
# # print("Image size:", size)

# # array = itk.array_from_image(image)
# # print("Array shape:", array.shape)  # (z, y, x)
# # tif.imwrite("/home/confetti/data/rm009/rm009_roi/mask.tiff",array)

#%%
import os
import tifffile
import numpy as np



def get_range_z_stack(z_start = 16176,z_end = 16300 ):
    # do not include z_end slice
    # Generate expected filenames
    filenames = [f"Z{z:05d}_C4.tif" for z in range(z_start, z_end )]
    folder = "/home/confetti/data/rm009/rm009_roi/4"
    # Load and stack into a 3D array
    volume_slices = []
    for fname in filenames:
        fpath = os.path.join(folder, fname)
        if os.path.exists(fpath):
            img = tifffile.imread(fpath)
            volume_slices.append(img)
        else:
            raise FileNotFoundError(f"{fname} not found in directory.")
    return volume_slices
#%%

# ######## padding the roi_with 24(96um) at both z slice######
volume_slices = get_range_z_stack(z_start = 16176,z_end = 16300)
# Stack into a 3D volume (Z, Y, X)
volume = np.stack(volume_slices, axis=0)
Z,Y,X = volume.shape
print("Volume shape:", volume.shape)  # (Z, Y, X)
print("Volume dtype:", volume.dtype)
tifffile.imwrite(f"/home/confetti/data/rm009/rm009_roi/z{z_start}_z{z_end}C4_d{Z}_h{Y}_w{X}.tif",volume)
#%%
import tifffile as tif
old_mask = tif.imread("/home/confetti/data/rm009/rm009_roi/single_layer/Z64800-65104_dilated_only_signalregion_mask_interpated_ord.tif")
mask = np.pad(old_mask,((24, 24), (0, 0), (0, 0)), mode='constant', constant_values=0)
Z,Y,X = mask.shape
tif.imwrite(f"/home/confetti/data/rm009/rm009_roi/single_dilated__ordered_mask_{Z}_{Y}_{X}.tiff",mask)


#%%
############# crop the big roi and mask into chunks, ready for dataloader training ############

import tifffile as tif
import numpy as np
import os
import tifffile


def crop_3d_with_stride_and_filter(image, mask, crop_size=(124, 1024, 1024), stride=512, threshold=0.08):
    D, H, W = image.shape
    d, h, w = crop_size

    assert D == d, "Depth of crop must match image depth"

    roi_list = []
    mask_list = []

    for y in range(0, H - h + 1, stride):
        for x in range(0, W - w + 1, stride):
            img_crop = image[:, y:y+h, x:x+w]
            mask_crop = mask[:, y:y+h, x:x+w]

            # Calculate ratio of non-zero voxels in the mask
            nonzero_ratio = np.count_nonzero(mask_crop) / mask_crop.size

            # Keep only if non-zero region > threshold
            if nonzero_ratio > threshold:
                roi_list.append(img_crop)
                mask_list.append(mask_crop)

    if len(roi_list) == 0:
        return np.empty((0, d, h, w), dtype=image.dtype), np.empty((0, d, h, w), dtype=mask.dtype)

    return np.stack(roi_list), np.stack(mask_list)


def save_rois_to_dirs(roi_images, roi_masks, roi_dir, mask_dir):
    os.makedirs(roi_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for i in range(roi_images.shape[0]):
        roi = roi_images[i]   # shape: (124, 1024, 1024)
        mask = roi_masks[i]   # shape: (124, 1024, 1024)

        roi_path = os.path.join(roi_dir, f"{i:04d}.tiff")
        mask_path = os.path.join(mask_dir, f"{i:04d}_mask.tiff")

        tifffile.imwrite(roi_path, roi)    # or uint16/float32 based on your data
        tifffile.imwrite(mask_path, mask.astype('uint8'))  # adjust dtype if needed

# Example usage:

roi = tif.imread("/home/confetti/data/rm009/rm009_roi/z16176_z16300C4_d124_h3500_w5250.tif")
mask = tif.imread("/home/confetti/data/rm009/rm009_roi/single_dilated_mask_124_3500_5250.tiff")

# image and mask shapes: (124, 3500, 5250)
roi_images, roi_masks = crop_3d_with_stride_and_filter(roi, mask)
print("Filtered ROI image shape:", roi_images.shape)
print("Filtered ROI mask shape:", roi_masks.shape)

# Save to directories
save_rois_to_dirs(
    roi_images, roi_masks,
    roi_dir="/home/confetti/data/rm009/boundary_seg/all_rois",
    mask_dir="/home/confetti/data/rm009/boundary_seg/all_masks",
)
# %%
###### extract feature ready for training the segmentation head
import sys
sys.path.append("/home/confetti/e5_workspace/hive1")

# %%
from lib.datasets.simple_dataset import get_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import tifffile as tif
import pickle
import os
from config.load_config import load_cfg
from lib.arch.ae import  load_compose_encoder_dict,build_final_model
device ='cuda'
args = load_cfg('config/seghead.yaml')
args.filters = [32,64]
args.mlp_filters =[64,32,24,12]
args.last_encoder = False

exp_name ='smallestv1roi_oridfar256_10000epoch'
feats_save_dir = f"/home/confetti/data/rm009/v1_roi1_seg_valid/l2_pool8_{exp_name}"
os.makedirs(feats_save_dir,exist_ok=True)

dataset = get_dataset(args)
loader = DataLoader(dataset,batch_size=1,drop_last=False,shuffle=False,num_workers=0)
E5 = False

if E5:
    data_prefix = "/share/home/shiqiz/data"
    workspace_prefix = "/share/home/shiqiz/workspace/hive"
else:
    data_prefix = "/home/confetti/data"
    workspace_prefix = '/home/confetti/e5_workspace/hive'

cmpsd_model = build_final_model(args)
cmpsd_model.eval().to(device)

# the latter conv_layer parameters will not be loaded
cnn_ckpt_pth = f'{data_prefix}/weights/rm009_3d_ae_best.pth'
mlp_ckpt_pth =f'{data_prefix}/weights/rm009_smallestv1roi_oridfar256_l2_pool8_10000.pth'
load_compose_encoder_dict(cmpsd_model,cnn_ckpt_pth,mlp_ckpt_pth,dims=args.dims)

from confettii.plot_helper import three_pca_as_rgb_image

for idx, img in enumerate(tqdm(loader)):
    feats = cmpsd_model(img.to(device))
    feats = np.squeeze(feats.cpu().detach().numpy())
    spatial_shape = feats.shape[1:]
    feats_lst = np.moveaxis(feats,0,-1)
    feats_lst = feats_lst.reshape(-1,feats.shape[0])
    rgb_img=three_pca_as_rgb_image(feats_lst,spatial_shape) 
    with open(f"{feats_save_dir}/{idx:04d}_feats.pkl",'wb') as f:
        pickle.dump(feats,f)
    tif.imwrite(f"{feats_save_dir}/{idx:04d}_rgb_feats.tif",rgb_img)


#%%
import tifffile as tif
from skimage import io
from pathlib import Path
from scipy.ndimage import zoom
import re

LAYER_ORDER = ["L1", "L23", "L4A", "L4B", "L4Ca", "L4Cb", "L5", "L6"]

def load_layer_masks(base_dir: str,
                     z_filename: str,
                     zoom_factor: float = None,
                     order: int = 0):
    """
    Load the same Z index mask from each cortical layer directory.

    Parameters
    ----------
    base_dir : str
        Path up to (and including) the parent of all per-layer directories.
        e.g. "/home/confetti/data/rm009/rm009_roi/single_layer"
    z_filename : str
        A filename that contains the target Z index, e.g. "L1-Z55200.png"
        (Only the numeric Z part is used; the 'L1-' prefix is ignored.)
    zoom_factor : float, optional
        If given, each mask is zoomed isotropically by this factor.
    order : int, default 0
        Interpolation order for `scipy.ndimage.zoom`. Use 0 for labels.

    Returns
    -------
    masks : list[np.ndarray]
        List of masks in the order defined by LAYER_ORDER.
        Missing files yield `None` in that position.
    missing : list[str]
        List of layer names that were not found.
    target_z : str
        The extracted Z index string (e.g. "55200").
    """
    base = Path(base_dir)
    # Extract the Z index (sequence of digits after '-Z')
    m = re.search(r'-Z(\d+)', z_filename)
    if not m:
        raise ValueError(f"Could not extract Z index from '{z_filename}'")
    target_z = m.group(1)

    masks = []
    missing = []

    for layer in LAYER_ORDER:
        # Expected filename pattern: <Layer>-Z<Z>.png
        fn = f"{layer}-Z{target_z}.png"
        path = base / layer / fn
        if path.is_file():
            mask = io.imread(path)
            if zoom_factor is not None:
                mask = zoom(mask, zoom_factor, order=order)
            masks.append(mask)
        else:
            masks.append(None)
            missing.append(layer)

    return masks, missing, int(target_z)

# --- Example usage ---
if __name__ == "__main__":
    base_dir = "/home/confetti/data/rm009/rm009_roi/single_layer"
    example_path = "/home/confetti/data/rm009/rm009_roi/single_layer/L1/L1-Z55200.png"

    # Just take the filename to extract Z
    z_filename = Path(example_path).name

    masks, missing, z = load_layer_masks(base_dir, z_filename, zoom_factor=2, order=0)

    print(f"Loaded Z index: {z}")
    print("Missing layers:", missing)
    for layer, arr in zip(LAYER_ORDER, masks):
        print(layer, None if arr is None else arr.shape)
#%%
z_start = int(int(z)/4)
roi = get_range_z_stack(z_start=z_start,z_end=z_start+1)
print(len(roi))
#%%
import napari
viewer  = napari.Viewer(ndisplay=2)
[viewer.add_image(img) for img in roi]
[viewer.add_labels(msk,opacity=0.5) for msk in masks]
napari.run()
#%%
from skimage import io
import numpy as np

example_path = "/home/confetti/data/rm009/rm009_roi/single_layer/L1/L1-Z55200.png"
mask = io.imread(example_path)
print(np.unique(mask))

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

# %%
