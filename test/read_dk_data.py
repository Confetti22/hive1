#%%
import tifffile as tif
img = tif.imread("/home/confetti/data/dk/MD594/downsampled/124.tif")
print(f"{img.shape= }")
import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
# plt.imshow(img[100:600,:])
plt.imshow(img)

#%%
import tifffile as tif
mask = tif.imread("/home/confetti/data/rm009/boundary_seg/new_boundary_seg_data/masks_valid/0007.tiff")

import matplotlib.pyplot as plt
plt.imshow(mask[0])