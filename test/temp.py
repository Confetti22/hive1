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
