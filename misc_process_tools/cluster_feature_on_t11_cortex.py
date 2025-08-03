#%%
import sys
import os
# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)
import tifffile as tif
from helper.image_reader import Ims_Image

ims_vol = Ims_Image('/home/confetti/e5_data/t1779/t1779.ims',channel=2)
