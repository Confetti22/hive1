#%%
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, segmentation, color
from skimage.measure import label, regionprops
import tifffile as tif
from confettii.plot_helper import get_boundary_via_erosion

mask = tif.imread('/home/confetti/data/t1779/test_data_part_brain/human_mask_0006.tif')
mask = mask[32]
img = tif.imread('/home/confetti/data/t1779/test_data_part_brain/0006.tif')
img = img[32]


boundaries = get_boundary_via_erosion(mask)
# Optional: visualize
plt.imshow(mask,cmap='tab10')
plt.imshow(boundaries, cmap='gray',alpha=0.5)
plt.title('Boundaries of Connected Components')
plt.axis('off')
plt.show()
# %%
from confettii.plot_helper import get_smooth_contours

import napari
viewer = napari.Viewer()
viewer.add_labels(labeled_mask, name='Labeled Mask')

smoothed_contours = get_smooth_contours(labeled_mask)
plt.imshow(mask, cmap='gray')
for sc in smoothed_contours:
    plt.plot(sc[:, 1], sc[:, 0], '-r', linewidth=0.5)
plt.axis('off')
plt.show()


# Add smoothed boundaries as shape paths
viewer.add_shapes(
    smoothed_contours,
    shape_type='path',
    edge_color='white',
    edge_width=5,
    name='Smoothed Boundaries'
)

napari.run()
# %%
