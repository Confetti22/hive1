#%%
import tifffile as tif
from pathlib import Path
import random
from tqdm import tqdm

eval_data_no_lst = [2,8,16,60,70,120] 
dir = '/home/confetti/e5_data/wide_filed/nuclei_channel'
raw_dir =Path(dir)
raw_fnames = sorted(list(raw_dir.rglob("*.tif")))
print(len(raw_fnames))

#%%
train_fnames = [fname for idx, fname in enumerate(raw_fnames) if idx not in eval_data_no_lst]
print(len(train_fnames))
eval_fnames = [raw_fnames[i] for i in eval_data_no_lst]
print(len(eval_fnames))
# %%
train_dir = f"/home/confetti/data/wide_filed/vsi_train"
eval_dir = f"/home/confetti/data/wide_filed/vsi_eval"

roi_size =  512 
roi_stride = 256
save_probliy = 0.125
saved_cnt = 0  # To track how many were actually saved

#%%
# eval the cropped img from the fist train_data:
import numpy as np
img = tif.imread(train_fnames[0])
img_shape = img.shape


margin = [int(x // 10) for x in img.shape]
roi_list = []

# Parameters (define if not already)
roi_size = 512       # example size
roi_stride = 256      # no overlap
# These can be adjusted

from confettii.entropy_helper import entropy_filter 
from skimage.measure import shannon_entropy
filter = entropy_filter(thres= 7)

# Calculate number of ROIs along each axis
x_steps = (img_shape[0] - 2 * margin[0] - roi_size) // roi_stride + 1
y_steps = (img_shape[1] - 2 * margin[1] - roi_size) // roi_stride + 1

# Initialize empty canvas
merged1 = np.zeros((x_steps * roi_size, y_steps * roi_size), dtype=img.dtype)
merged2 = np.zeros((x_steps * roi_size, y_steps * roi_size), dtype=img.dtype)

for idx, x_offset in enumerate(range(margin[0], img_shape[0] - roi_size - margin[0] + 1, roi_stride)):
    for idy, y_offset in enumerate(range(margin[1], img_shape[1] - roi_size - margin[1] + 1, roi_stride)):
        roi = img[x_offset:x_offset+roi_size, y_offset:y_offset+roi_size]
        entrop =  shannon_entropy(roi)
        template_roi = np.ones(shape=(roi_size,roi_size))
        merged1[idx * roi_size : (idx+1) * roi_size,
               idy * roi_size : (idy+1) * roi_size] = roi
        merged2[idx * roi_size : (idx+1) * roi_size,
               idy * roi_size : (idy+1) * roi_size] = template_roi* entrop

import napari 
viewer = napari.Viewer()
viewer.add_image(merged1)
viewer.add_image(merged2)
napari.run()
# Save or show
#%%
from confettii.entropy_helper import entropy_filter 
filter = entropy_filter(thres= 7)
for fname in tqdm(train_fnames):
    img = tif.imread(fname)
    img_shape = img.shape
    margin = [int(x//10) for x in img.shape]
    for x_offset in range(margin[0], img_shape[0] - roi_size - margin[0], roi_stride):
        for y_offset in range(margin[1], img_shape[1] - roi_size - margin[1], roi_stride):
            roi = img[x_offset:x_offset+roi_size, y_offset:y_offset+roi_size]
            if filter(roi):
                if random.random() < save_probliy:  # 25% probability
                    tif.imwrite(f"{train_dir}/{saved_cnt+1:05d}.tif", roi)
                    saved_cnt += 1
print('for training data: total saved:', saved_cnt)

saved_cnt = 0
for fname in tqdm(eval_fnames):
    img = tif.imread(fname)
    img_shape = img.shape
    margin = [int(x//10) for x in img.shape]
    for x_offset in range(margin[0], img_shape[0] - roi_size - margin[0], roi_stride):
        for y_offset in range(margin[1], img_shape[1] - roi_size - margin[1], roi_stride):
            roi = img[x_offset:x_offset+roi_size, y_offset:y_offset+roi_size]
            if filter(roi):
                if random.random() < save_probliy:  # 25% probability
                    tif.imwrite(f"{eval_dir}/{saved_cnt+1:05d}.tif", roi)
                    saved_cnt += 1
print('for eval data: total saved:', saved_cnt)


# %%
