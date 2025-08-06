#%%
import sys
import os
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)
from helper.image_reader import Ims_Image
import napari
viewer = napari.Viewer()

path ='/home/confetti/mnt/data/VISoR_Reconstruction/SIAT_SIAT/BiGuoqiang/Macaque_Brain/RM009_2/RM009_all_/009.ims'
for channel in [1]:
    ims_vol = Ims_Image(path,channel=channel)
    Z,Y,X = ims_vol.rois[0][3:]
    z_slice = ims_vol.from_roi(coords=(13750,3500,9250,1,3500,5250))
    # z_slice = ims_vol.from_roi(coords=(int(Z//2+30),0,0,1,Y,X))
    viewer.add_image(z_slice,name=f"channel_{channel}", contrast_limits=(0,6000))
    del ims_vol

napari.run()
# %%
