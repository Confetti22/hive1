from helper.image_reader import Ims_Image
from skimage.measure import shannon_entropy
import tifffile as tif
import numpy as np
from scipy.ndimage import zoom
import os
import napari
from pprint import pprint

def entropy_filter(l_thres=1.4, h_thres=100):
    def _filter(img):
        entropy=shannon_entropy(img)
        if (entropy>= l_thres) and (entropy <= h_thres):
            print(f"entrop of the roi is {entropy}")
            return True
        else:
            return False
    return _filter

filter = entropy_filter(l_thres=2.7)


#read small cube from ims
channel = 0
level = 0
image_path = "/home/confetti/e5_data/rm009/rm009.ims"
ims_vol = Ims_Image(image_path, channel=channel)
vol_shape = ims_vol.info[level]['data_shape']
pprint(f"{ims_vol.info}")
import napari
from dask_image.imread import imread
#read whole slice from stack of tif, utilizing lazing loading
stack = imread("/home/confetti/mnt/data/VISoR_Reconstruction/SIAT_SIAT/BiGuoqiang/Macaque_Brain/RM009_2/RM009_all_/ROIImage/4.0/*C1.tif")
viewer = napari.Viewer()
image_layer = viewer.add_image(stack,contrast_limits=[0,4000],multiscale=False)
# napari.view_image(stack1, contrast_limits=[0,4000], multiscale=False)


@viewer.mouse_double_click_callbacks.append
def on_double_click(viewer, event):
    # Get the coordinate of click point -- P_i
    mouse_pos = viewer.cursor.position
    P_i = image_layer.world_to_data(mouse_pos)
    P_i = np.round(P_i).astype(int)  # Convert to int indices

    roi_size = (64, 64, 64)  # (z, y, x)
    half_size = np.array(roi_size) // 2
    image_shape = image_layer.data.shape

    # Calculate the start and end indices of the ROI
    start = P_i - half_size
    end = P_i + half_size

    # Clip the indices to stay within the image boundaries
    start_clipped = np.maximum(start, 0)
    end_clipped = np.minimum(end, image_shape)
    final_size = end_clipped - start_clipped

    coords = np.concatenate((start_clipped,final_size))
    # Extract the clipped ROI from image data
    roi = ims_vol.from_roi(coords=coords,level=0)
    res = shannon_entropy(roi)
    # res = filter(roi)
    print(f"entrop is:{res} at pos:{coords}")
napari.run()
