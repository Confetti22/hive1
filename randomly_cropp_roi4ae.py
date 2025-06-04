from helper.image_reader import Ims_Image
from skimage.measure import shannon_entropy
import tifffile as tif
import numpy as np
from scipy.ndimage import zoom
import os

def entropy_filter(l_thres=1.4, h_thres=100):
    def _filter(img):
        entropy=shannon_entropy(img)
        if (entropy>= l_thres) and (entropy <= h_thres):
            # print(f"entrop of the roi is {entropy}")
            return True
        else:
            return False
    return _filter


save_dir="/home/confetti/data/rm009/valid_roi"
os.makedirs(save_dir,exist_ok=True)


image_path = "/home/confetti/e5_data/rm009/rm009.ims"
level = 0
channel = 0
roi_size =(64,64,64)
amount = 128
cnt = 1
sample_range = [[1000,17800],[100,15000],[100,16000]]

ims_vol = Ims_Image(image_path, channel=channel)
vol_shape = ims_vol.info[level]['data_shape']

import time
current = time.time()
while cnt < amount:

    roi,indexs=ims_vol.get_random_roi(filter=entropy_filter(l_thres=2.7),roi_size=roi_size,level=0,skip_gap =False,sample_range=sample_range)
    file_name = f"{save_dir}/{cnt:04d}.tif"
    tif.imwrite(file_name,roi)
    # print(f"{file_name} has been saved ")
    cnt = cnt +1

print(f"fininshed: time:{time.time() - current}")

