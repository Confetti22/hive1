#!/usr/bin/env python3
import re
import argparse
from pathlib import Path
import sys
import tifffile as tif
import numpy as np
import os

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

def crop_3d_with_stride_and_filter(image, mask, crop_size=(124, 1024, 1024), stride=512, threshold=0.08):
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
        mask_path = os.path.join(mask_dir, f"{gloable_idx:04d}_mask.tiff")

        tif.imwrite(roi_path, roi)    # or uint16/float32 based on your data
        tif.imwrite(mask_path, mask.astype('uint8'))  # adjust dtype if needed


def main1():
    """
    collect correspond z_slice to merge into 32-depth volume and then crop into rois and filter based on mask-nonzero percentage
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
        roi_list, mask_list = crop_3d_with_stride_and_filter(stack,mask=mask[None,...],crop_size=[32,1024,1024],stride=512,threshold=0.08)
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

def main2():
    global LAST_CNT,CNT
    ap = argparse.ArgumentParser(description="Strict 4µm slice stacking around each 1µm mask z index.")
    ap.add_argument("--mask_dir", default="/home/confetti/data/rm009/boundary_seg/ori_mask_valid",  type=Path)
    ap.add_argument("--img_dir",  default="/home/confetti/data/rm009/boundary_seg/ori_img_stack_valid",  type=Path)
    args = ap.parse_args()


    img_files = sorted(
                    [os.path.join(args.img_dir, fname) 
                    for fname in os.listdir(args.img_dir) 
                    if fname.endswith('.tif')])

    mask_files = sorted(
                    [os.path.join(args.mask_dir, fname) 
                    for fname in os.listdir(args.mask_dir) 
                    if fname.endswith('.tif')])

    save_roi_dir ="/home/confetti/data/rm009/boundary_seg/valid_rois"
    save_mask_dir ="/home/confetti/data/rm009/boundary_seg/valid_masks"
    for img_path, mask_path in tqdm(zip(img_files,mask_files), desc="Masks"):
        img = tif.imread(img_path)
        mask = tif.imread(mask_path)
        base = Path(mask_path).stem  # merged-Zddddd

        LAST_CNT = CNT
        roi_list, mask_list = crop_3d_with_stride_and_filter(img,mask=mask,crop_size=[32,1024,1024],stride=512,threshold=0.08)
        save_rois_to_dirs(roi_list,mask_list,roi_dir=save_roi_dir,mask_dir=save_mask_dir)
        print(f"{base}finished")
        

if __name__ == "__main__":
    main2()