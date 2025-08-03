#%%
import sys
import os
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)
from helper.image_reader import Ims_Image
import napari
viewer = napari.Viewer()

path ='/home/confetti/mnt/data/VISoR_Reconstruction/SIAT_SIAT/BiGuoqiang/Macaque_Brain/RM009_2/RM009_all_/009.ims'
for channel in range(2):
    ims_vol = Ims_Image(path,channel=channel)
    Z,Y,X = ims_vol.rois[0][3:]
    print(Z,Y,X)
    z_slice = ims_vol.from_roi(coords=(9000,3000,9000,1,6000,5000))
    viewer.add_image(z_slice,name=f"channel_{channel}", contrast_limits=(0,6000))
    del ims_vol

napari.run()
# %%
