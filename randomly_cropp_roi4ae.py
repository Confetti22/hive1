from helper.image_reader import Ims_Image
from skimage.measure import shannon_entropy
import tifffile as tif
import numpy as np
from scipy.ndimage import zoom
import os
import random

def entropy_filter(l_thres=1.4, h_thres=100):
    def _filter(img):
        entropy=shannon_entropy(img)
        if (entropy>= l_thres) and (entropy <= h_thres):
            # print(f"entrop of the roi is {entropy}")
            return True
        else:
            return False
    return _filter

def generate_sphereshell__shifts(R, r=0, dims=3):
    """Generate integer shifts within a sphere shell (radius R, inner radius r)."""
    shifts = []
    ranges = [range(-R, R+1)] * dims
    for shift in np.array(np.meshgrid(*ranges)).T.reshape(-1, dims):
        norm = np.linalg.norm(shift)
        if r < norm <= R:
            shifts.append(shift)
    return np.array(shifts)


def positve_pair_loc_generate(loc,all_near_shifts):
    shift = random.choice(all_near_shifts)
    return loc+shift



knn = 4
save_exp_name= '64_roi'
save_parent_dir="/home/confetti/e5_data/t1779"
save_dirs = []
for i in range(knn):
    dir = f"{save_parent_dir}/{save_exp_name}_{i+1}"
    save_dirs.append(dir)
    os.makedirs(dir,exist_ok=True)

d_near = 64



ims_path = "/home/confetti/mnt/data/VISoR_Reconstruction/SIAT_SIAT/BiGuoqiang/Mouse_Brain/20210131_ZSS_USTC_THY1-YFP_1779_1/Reconstruction_1.0/z00000_c1.ims.part"
level = 0
channel = 2
roi_size =(64,64,64)
amount = 2**15 
cnt = 1

ims_vol = Ims_Image(ims_path, channel=channel)
D,H,W= ims_vol.rois[level][3:]
margin = 10 

# Generate random (x, y, z) locations within the given range
lz, hz = d_near + margin +int(D//4),  int(D*3/4) - d_near - margin
ly, hy = d_near + margin +int(H//4),  int(H*3/4) - d_near - margin
lx, hx = d_near + margin +int(W//4),  int(W//2) - d_near - margin

# sample_range = [[1000,17800],[100,15000],[100,16000]] #ae sample range for rm009
sample_range = [[lz,hz], [ly,hy],[lx,hx]]

vol_shape = ims_vol.info[level]['data_shape']


all_near_shifts = generate_sphereshell__shifts(R= d_near,r= 24 ,dims=3)

import time
current = time.time()
while cnt < amount:

    roi,indexs=ims_vol.get_random_roi(filter=entropy_filter(l_thres=1.7),roi_size=roi_size,level=0,skip_gap =False,sample_range=sample_range)


    pair_locs = [positve_pair_loc_generate(indexs,all_near_shifts) for _ in range(knn -1)]
    pair_roi = [ims_vol.from_roi(coords=(*pair_loc,64,64,64)) for pair_loc in pair_locs]
    pair_roi.insert(0,roi)

    for dir, roi in zip(save_dirs, pair_roi):
        file_name = f"{dir}/{cnt:06d}.tif"
        tif.imwrite(file_name,roi)

    cnt = cnt +1
    print(f"{cnt} and neighbours has been saved ")

print(f"fininshed: time:{time.time() - current}")

