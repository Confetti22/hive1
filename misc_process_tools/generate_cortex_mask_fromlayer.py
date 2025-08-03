import os
import numpy as np
from PIL import Image
import tifffile as tif
from scipy.ndimage import binary_dilation
from skimage.morphology import disk  # for 2D dilation footprint
from tqdm import tqdm

def dilate_masks(input_dir, output_dir, dilation_radius=8):
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
        

if __name__ == '__main__':
    input_directory = '/home/confetti/data/rm009/boundary_seg/layer_masks'
    output_directory = '/home/confetti/data/rm009/boundary_seg/masks'
    dilate_masks(input_directory, output_directory)