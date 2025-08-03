"""
image --> predict: downsample by 1/4 --> avg8pooling, pad=0,stride=1;
predict --> image: upsample by 4 --> padding 14 at both side
"""
#%%
import numpy as  np
import os
from scipy.ndimage import zoom
from pprint import pprint
def upsample2imagespace(pred):
    """
    pred: H*W*3, rgb image;
    corresponding roi_image: 32*1024*1024;
    """
    pred = pred[None,...] # (D,W,W,C)
    zoomed= zoom(pred,zoom=(4,4,4,1),order=0)
    padded = np.pad(zoomed,pad_width=((14,14),(14,14),(14,14),(0,0)),mode='constant')
    return padded

input_dir ='/home/confetti/e5_workspace/hive1/outs/seg_bnd/bnd_seg_finetune_scratch_3moduler_level2_avg_pool8_cldice_smallerlr/seg_valid_result_images'
fnames = sorted([os.path.join(input_dir,fname) for fname in os.listdir(input_dir) if fname.endswith('560.tiff')])
pprint(fnames)
    
# %%
import tifffile as tif
from pathlib import Path
save_dir ='/home/confetti/data/rm009/boundary_seg/temp'
for path in fnames:
    pred = tif.imread(path)
    fname = Path(path).stem
    padded = upsample2imagespace(pred)
    tif.imwrite(f"{save_dir}/{fname}_traing_scratch.tiff",padded)
# %%
