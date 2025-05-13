#%%
import os
os.environ["NAPARI_ASYNC"] = "1"

import napari
from skimage.io import imread
from skimage.io.collection import alphanumeric_key
from dask import delayed
import dask.array as da
from pathlib import Path
from glob import glob
# 
# directory = Path("/home/confetti/mnt/data/VISoR_Reconstruction/SIAT_SIAT/BiGuoqiang/Mouse_Brain/20210131_ZSS_USTC_THY1-YFP_1779_1/Reconstruction_1.0/Reconstruction/BrainImage/1.0")
# directory = Path("/home/confetti/mnt/data/VISoR_Reconstruction/SIAT_SIAT/BiGuoqiang/Mouse_Brain/20210131_ZSS_USTC_THY1-GFP_11_1/Reconstruction/backup/BrainImage/4.0")
directory = Path("/home/confetti/mnt/data/VISoR_Reconstruction/SIAT_SIAT/BiGuoqiang/Macaque_Brain/RM009_2/Analysis/ROIReconstruction/ROIImage/4.0")
# directory = Path("/home/confetti/mnt/data/VISoR_Reconstruction/SIAT_SIAT/BiGuoqiang/Macaque_Brain/RM009_2/RM009_all_/ROIImage/4.0")

# Find all files ending with 'C4.tif'
filenames = sorted(list(directory.rglob("*C4.tif")))
print(len(filenames),'\n', filenames[-1])

# read the first file to get the shape and dtype
# ASSUMES THAT ALL FILES SHARE THE SAME SHAPE/TYPE
sample = imread(filenames[0])
print(f"slice shape {sample.shape}")

lazy_imread = delayed(imread)  # lazy reader
lazy_arrays = [lazy_imread(fn) for fn in filenames]
dask_arrays = [
    da.from_delayed(delayed_reader, shape=sample.shape, dtype=sample.dtype)
    for delayed_reader in lazy_arrays
]
# Stack into one large dask.array
stack = da.stack(dask_arrays, axis=0)

print(stack.shape ) # (nfiles, nz, ny, nx)


# %%
import napari
from dask_image.imread import imread

# stack = imread("/home/confetti/e5_data/wide_filed/nuclei_channel/*.tif")

viewer = napari.Viewer()
viewer.add_image(stack,contrast_limits=[0,4000],multiscale=False)
# napari.view_image(stack1, contrast_limits=[0,4000], multiscale=False)

napari.run()

# %%
