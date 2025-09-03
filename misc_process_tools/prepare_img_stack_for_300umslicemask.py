#%%
#!/usr/bin/env python3
import re
import argparse
from pathlib import Path
import sys
from glob import glob
from pathlib import Path
import tifffile as tif
import numpy as np
import os
import numpy as np
from PIL import Image
import tifffile as tif
from scipy.ndimage import binary_dilation
from skimage.morphology import disk  # for 2D dilation footprint
from tqdm import tqdm
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **k: x   # fallback no-op

MASK_RE = re.compile(r'^merged-Z(\d{5})_2x\.tif$')
IMG_TEMPLATE = "Z{z:05d}_C4.tif"
Z_BEFORE = 16
Z_AFTER  = 15
TOTAL_SLICES = Z_BEFORE + Z_AFTER + 1  # 32
CNT =0
LAST_CNT=0

def find_masks(mask_dir: Path):
    items = []
    for p in mask_dir.iterdir():
        m = MASK_RE.match(p.name)
        if m:
            z = int(m.group(1))
            items.append((z, p))
    items.sort(key=lambda t: t[0])
    print(len(items))
    return items

def build_stack(z_idx_1um: int, img_dir: Path):
    z_idx_4 = z_idx_1um // 4
    z_range = range(z_idx_4 - Z_BEFORE, z_idx_4 + Z_AFTER + 1)
    slices = []
    missing = []
    for z in z_range:
        fname = IMG_TEMPLATE.format(z=z)
        fpath = img_dir / fname
        if not fpath.is_file():
            missing.append(fname)
        else:
            img = tif.imread(fpath)
            if img.ndim != 2:
                raise ValueError(f"{fpath} not 2D (shape={img.shape})")
            slices.append(img)
    if missing:
        return None, list(z_range), missing
    stack = np.stack(slices, axis=0)  # (32,H,W)
    return stack, list(z_range), []

def crop_3d_with_stride_and_filter_out_sparse_one(image, mask, crop_size=(124, 1024, 1024), stride=512, threshold=0.08):
    """
    generate candidate rois at given stride and roi_size, then filter out rois with mask that has too many background percentage
    """
    global CNT
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
                CNT +=1
                roi_list.append(img_crop)
                mask_list.append(mask_crop)

    if len(roi_list) == 0:
        return np.empty((0, d, h, w), dtype=image.dtype), np.empty((0, d, h, w), dtype=mask.dtype)

    return np.stack(roi_list), np.stack(mask_list)

def save_rois_to_dirs(roi_images, roi_masks, roi_dir, mask_dir):
    os.makedirs(roi_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for i in range(roi_images.shape[0]):
        gloable_idx = LAST_CNT+1+i
        roi = roi_images[i]   # shape: (124, 1024, 1024)
        mask = roi_masks[i]   # shape: (124, 1024, 1024)

        roi_path = os.path.join(roi_dir, f"{gloable_idx:04d}.tiff")
        mask_path = os.path.join(mask_dir, f"{gloable_idx:04d}.tiff")

        tif.imwrite(roi_path, roi)    # or uint16/float32 based on your data
        tif.imwrite(mask_path, mask.astype('uint8'))  # adjust dtype if needed


def step1_assemble_into_stacks():
    """
    collect correspond z_slice to merge into 32-depth volume 
    """
    global LAST_CNT,CNT
    ap = argparse.ArgumentParser(description="Strict 4µm slice stacking around each 1µm mask z index.")
    ap.add_argument("--mask_dir", default="/home/confetti/data/rm009/rm009_roi/single_layer/merged_bnds/upscaled_before_smooth/masks",  type=Path)
    ap.add_argument("--img_dir",  default="/home/confetti/mnt/data/VISoR_Reconstruction/SIAT_SIAT/BiGuoqiang/Macaque_Brain/RM009_2/Analysis/ROIReconstruction/ROIImage/4.0",  type=Path)
    ap.add_argument("--out_dir",  default="/home/confetti/data/rm009/boundary_seg",  type=Path)
    ap.add_argument("--suffix", default="_stack", help="Suffix appended to mask base name.")
    ap.add_argument("--log_name", default="missing_stacks.log")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    masks = find_masks(args.mask_dir)
    if not masks:
        print(f"No mask files in {args.mask_dir}", file=sys.stderr)
        sys.exit(1)

    missing_logs = []
    made = 0
    save_roi_dir ="/home/confetti/data/rm009/boundary_seg/rois"
    save_mask_dir ="/home/confetti/data/rm009/boundary_seg/masks"
    for z_idx, mask_path in tqdm(masks, desc="Masks"):
        stack, z_range, missing = build_stack(z_idx, args.img_dir)
        base = mask_path.stem  # merged-Zddddd
        if stack is None:
            missing_logs.append(f"{base}: MISSING {len(missing)} / {TOTAL_SLICES} -> {missing}")
            continue
        mask = tif.imread(mask_path)
        LAST_CNT = CNT
        roi_list, mask_list = crop_3d_with_stride_and_filter_out_sparse_one(stack,mask=mask[None,...],crop_size=[32,1024,1024],stride=512,threshold=0.08)
        save_rois_to_dirs(roi_list,mask_list,roi_dir=save_roi_dir,mask_dir=save_mask_dir)

        out_name = f"{base}{args.suffix}.tif"
        out_path = args.out_dir / out_name
        tif.imwrite(out_path, stack, dtype=stack.dtype)
        made += 1

    # Summary
    print(f"\nCreated {made} stacks. Skipped {len(missing_logs)}.")

    if missing_logs:
        log_path = args.out_dir / args.log_name
        with open(log_path, "w") as f:
            f.write("\n".join(missing_logs))
        print(f"Missing details written to: {log_path}")






def step2_crop_stacks():
    global LAST_CNT,CNT
    ap = argparse.ArgumentParser(description="Strict 4µm slice stacking around each 1µm mask z index.")
    ap.add_argument("--mask_dir",   type=Path)
    ap.add_argument("--img_dir",    type=Path)
    args = ap.parse_args()
    valid_flag = False 
    parent_dir ="/home/confetti/data/rm009/boundary_seg/ori_stack_mask"
    args.img_dir = f"{parent_dir}/ori_img_stack_valid" if valid_flag else f"{parent_dir}/ori_img_stack"
    args.mask_dir = f"{parent_dir}/ori_mask_valid" if valid_flag else f"{parent_dir}/ori_mask"


    img_files = sorted(
                    [os.path.join(args.img_dir, fname) 
                    for fname in os.listdir(args.img_dir) 
                    if fname.endswith('.tif')])

    mask_files = sorted(
                    [os.path.join(args.mask_dir, fname) 
                    for fname in os.listdir(args.mask_dir) 
                    if fname.endswith('.tif')])

    save_parent_dir ="/home/confetti/data/rm009/boundary_seg/new_boundary_seg_data"
    save_roi_dir =f"{save_parent_dir}/rois_valid" if valid_flag else f"{save_parent_dir}/rois" 
    save_mask_dir =f"{save_parent_dir}/masks_valid" if valid_flag else f"{save_parent_dir}/masks"
    os.makedirs(save_roi_dir,exist_ok=True)
    os.makedirs(save_mask_dir,exist_ok=True)
    for img_path, mask_path in tqdm(zip(img_files,mask_files), desc="Masks"):
        img = tif.imread(img_path)
        mask = tif.imread(mask_path)
        base = Path(mask_path).stem  # merged-Zddddd

        LAST_CNT = CNT
        roi_list, mask_list = crop_3d_with_stride_and_filter_out_sparse_one(img,mask=mask,crop_size=[32,512,512],stride=256,threshold=0.4)
        save_rois_to_dirs(roi_list,mask_list,roi_dir=save_roi_dir,mask_dir=save_mask_dir)
        print(f"{base}finished")






def step3_dilate_layer_mask_into_cortex_mask(input_dir, output_dir, dilation_radius=8):
    """
    Applies morphological dilation to all mask files in a specified input directory
    and saves the results to an output directory.

    This script is intended to be run as a standalone program. It uses the
    `dilate_masks` function to process each file in the input directory.

    To run the script on a specific set of mask files:

    input_directory = '/home/confetti/data/rm009/boundary_seg/layer_masks'
    output_directory = '/home/confetti/data/rm009/boundary_seg/masks'
    dilate_masks(input_directory, output_directory)
    """

    os.makedirs(output_dir, exist_ok=True)

    dir_list = sorted([os.path.join(input_dir,fname) for fname in os.listdir(input_dir) if fname.endswith(('.tiff','.tif'))])
    for idx, filename in enumerate(tqdm(dir_list)):
        if not filename.lower().endswith(('.tif', '.tiff')):
            continue

        # Load image
        img_path = os.path.join(input_dir, filename)
        img = tif.imread(img_path)
        img_np = np.array(img)
        img_np = np.squeeze(img)

        # Convert to binary mask (any non-zero becomes 1)
        binary_mask = img_np > 0

        # Dilation with disk-shaped footprint
        footprint = disk(dilation_radius)
        dilated_mask = binary_dilation(binary_mask, structure=footprint)

        # Save result
        tif.imwrite(f"{output_dir}/{idx+1:04d}.tif",dilated_mask)


import numpy as np
from scipy.ndimage import gaussian_filter

#typical sigma is set to be 0.8*stride to avoid aliasing while preserving as much useful signal as possible
def blur_and_downsample(volume, sigma=1.6, stride=2, num_stages=3):
    for _ in range(num_stages):
        volume = gaussian_filter(volume, sigma=sigma)
        volume = volume[::stride, ::stride, ::stride]
    return volume
        
import matplotlib.pyplot as plt


# %%

## downsample the roi_volumes(3d) and save the middle_slice(2d) for low-resolution reconstruction
def blur_and_down_sample_for_reconstruct_target(input_dir, output_dir, sigma=1.6, stride=2, num_stages=3, crop_size=63):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiff_paths = sorted(glob(str(input_dir / "*.tiff")))

    for idx, file_path in enumerate(tiff_paths):
        volume = tif.imread(file_path)

        smoothed = blur_and_downsample(volume.astype(np.float32), sigma=sigma, stride=stride, num_stages=num_stages)
        # Average two middle slices
        z = smoothed.shape[0] // 2
        avg = ((smoothed[z-1] + smoothed[z]) / 2).astype(np.float32)
        avg_uint16 = np.clip(np.round(avg), 0, 65535).astype(np.uint16)
        cropped = avg_uint16[:crop_size, :crop_size]

        save_path = output_dir / f"{idx +1 :04d}.tiff"
        tif.imwrite(save_path, cropped)
        print(f"Saved: {save_path}")

# Example usage
blur_and_down_sample_for_reconstruct_target(
    input_dir="/home/confetti/data/rm009/boundary_seg/new_boundary_seg_data/rois_valid",
    output_dir="/home/confetti/data/rm009/boundary_seg/new_boundary_seg_data/low_resol_rois_valid",
    sigma=1.6,
    stride=2,
    num_stages=3
)
# %%
