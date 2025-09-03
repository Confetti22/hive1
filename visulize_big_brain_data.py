#%%
import nibabel as nib
import numpy as np

# Load .mnc file
img = nib.load("/home/confetti/data/big_brian_annotation/s2807/geo_2807.mnc")
pm = nib.load("/home/confetti/data/big_brian_annotation/s2807/pm2807_nl_classifiedsixlayers_aligned.mnc")
raw = nib.load("/home/confetti/data/big_brian_annotation/s2807/raw_2807.mnc")

# Get data as NumPy array
data = img.get_fdata()
data = np.squeeze(data) 
print(f"Shape: {data.shape}")
print(f"Data type: {data.dtype}")

pm = pm.get_fdata()
pm = np.squeeze(pm)
print(f"{pm.shape= }")

raw = raw.get_fdata()
raw = np.squeeze(raw)
print(f"{raw.shape= }")

import napari
viewer = napari.Viewer()
viewer.add_image(data)
viewer.add_image(pm)
viewer.add_image(raw)
napari.run()
# %%
