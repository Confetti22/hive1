#%%
import dask.array as da
import zarr
from napari_lazy_openslide import OpenSlideStore
#%%
store = OpenSlideStore('/home/confetti/e5_data/t1779/r5_c2_filtered.tif')
grp = zarr.open(store, mode="r")
#%%
# The OpenSlideStore implements the multiscales extension
# https://forum.image.sc/t/multiscale-arrays-v0-1/37930
datasets = grp.attrs["multiscales"][0]["datasets"]

pyramid = [grp.get(d["path"]) for d in datasets]
print(pyramid)
# [
#   <zarr.core.Array '/0' (23705, 29879, 4) uint8 read-only>,
#   <zarr.core.Array '/1' (5926, 7469, 4) uint8 read-only>,
#   <zarr.core.Array '/2' (2963, 3734, 4) uint8 read-only>,
# ]

pyramid = [da.from_zarr(store, component=d["path"]) for d in datasets]
print(pyramid)
# [
#   dask.array<from-zarr, shape=(23705, 29879, 4), dtype=uint8, chunksize=(512, 512, 4), chunktype=numpy.ndarray>,
#   dask.array<from-zarr, shape=(5926, 7469, 4), dtype=uint8, chunksize=(512, 512, 4), chunktype=numpy.ndarray>,
#   dask.array<from-zarr, shape=(2963, 3734, 4), dtype=uint8, chunksize=(512, 512, 4), chunktype=numpy.ndarray>,
# ]

# Now you can use numpy-like indexing with openslide, reading data into memory lazily!
low_res = pyramid[-1][:]
region = pyramid[0][y_start:y_end, x_start:x_end]
#%%
import pickle
import matplotlib.pyplot as plt
from confettii.plot_helper import grid_plot_list_imgs
with open('/home/confetti/data/wide_filed/test_hp_raw_feats.pkl','rb')as file:
    feats_lst = pickle.load(file)
print(feats_lst.shape)

lenght = int(feats_lst.shape[0]**0.5)
feats_map = feats_lst.reshape(lenght,lenght,feats_lst.shape[-1])
print(feats_map.shape)
plt.imshow(feats_map.std(axis=-1))
#%%
import SimpleITK as sitk

# Load the image
image = sitk.ReadImage('/home/confetti/data/rm009/rm009_roi/all-z64800-65104/All-Z64800-65104.mha')
print("Size:", image.GetSize())          # (x, y, z)
#%%
save_path = '/home/confetti/data/rm009/seg_valid'

# Define starting index and size for cropping
start = [1190,776,0]            # starting at origin
size = [1536,1536,64]          # ROI size

# Crop the image
roi = sitk.RegionOfInterest(image, size=size, index=start)
# Save as a multipage TIFF file (3D volume)
sitk.WriteImage(roi, f"{save_path}/001_cortex_mask.tiff")
# %%
import tifffile as tif
img = tif.imread("/home/confetti/data/rm009/rm009_roi/z16200_z16276C4.tif")
roi = img[0:64, 776:776+1536, 1190:1190+1536]
tif.imwrite(f"{save_path}/001_cortex.tiff",roi)


# %%

import os
import re
import time
current = time.time()
for i in [2,3,4]:
    dir_a = f'/home/confetti/e5_data/t1779/knn/64_roi_{i}'  # <-- Change this to your directory path

    for fname in os.listdir(dir_a):
        if fname.endswith('.tif'):
            base, ext = os.path.splitext(fname)
            if re.fullmatch(r'\d{4}', base):
                new_name = f"00{base}{ext}"
            elif re.fullmatch(r'\d{5}', base):
                new_name = f"0{base}{ext}"
            else:
                continue  # skip files that don't match the expected pattern
            old_path = os.path.join(dir_a, fname)
            new_path = os.path.join(dir_a, new_name)
            print(f"Renaming {fname} -> {new_name}")
            os.rename(old_path, new_path)
print(f"finished! time: {time.time()-current}")


# %%
