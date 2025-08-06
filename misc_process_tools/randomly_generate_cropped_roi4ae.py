import sys
import os
# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

from helper.image_reader import Ims_Image
from skimage.measure import shannon_entropy
import tifffile as tif
import os

def entropy_filter(l_thres=1.4, h_thres=100):
    def _filter(img):
        entropy=shannon_entropy(img)
        if (entropy>= l_thres) and (entropy <= h_thres):
            print(f"entrop of the roi is {entropy}")
            return True
        else:
            return False
    return _filter


save_dir="/home/confetti/data/rm009/roi_v1roi1_brain4ae_nissl_valid"
os.makedirs(save_dir,exist_ok=True)


image_path = "/home/confetti/e5_data/rm009/rm009.ims"
level = 0
channel = 1
roi_size =(64,64,64)
amount = 1024
cnt = 1
sample_range = [[13750,13750+3750],[3500,3500+3500],[4000,4000+5250]] #[[lz,hz],[ly,hy],[lx,hx]]

ims_vol = Ims_Image(image_path, channel=channel)
vol_shape = ims_vol.info[level]['data_shape']

while cnt < amount:

    roi,indexs=ims_vol.get_random_roi(filter=entropy_filter(l_thres=2.7),roi_size=roi_size,level=0,skip_gap =False,sample_range=sample_range,margin=0)
    file_name = f"{save_dir}/{cnt:04d}.tif"
    tif.imwrite(file_name,roi)
    print(f"{file_name} has been saved ")
    cnt = cnt +1

