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
import tifffile as tif
import numpy as np

mask_path = "/home/confetti/data/rm009/boundary_seg/bnd_masks/0001_mask_bnd_thick.tif"
mask = tif.imread(mask_path)
mask = np.squeeze(mask)
print(f"{mask.shape = }")
roi_path = "/home/confetti/data/rm009/boundary_seg/rois/0001.tiff"
roi = tif.imread(roi_path)
print(f"{roi.shape=}")
# %%
from  scipy.ndimage import zoom
ds_mask = zoom(mask,0.25,order=0)
ds_roi = zoom(roi,0.25,order=0)
import napari
viewer  = napari.Viewer()
viewer.add_image(ds_roi)
viewer.add_labels(ds_mask,opacity=0.5)
napari.run()

# %%
