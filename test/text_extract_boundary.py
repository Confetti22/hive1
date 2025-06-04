#%%
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, segmentation, color
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion
import tifffile as tif

mask = tif.imread('/home/confetti/data/t1779/test_data_part_brain/human_mask_0006.tif')
mask = mask[32]
img = tif.imread('/home/confetti/data/t1779/test_data_part_brain/0006.tif')
img = img[32]


# Step 1: Label the connected components if binary
labeled_mask = label(mask)

# Step 2: Extract boundaries of each region
boundaries = np.zeros_like(labeled_mask, dtype=bool)

for region_label in np.unique(labeled_mask):
    if region_label == 0:
        continue  # skip background

    # Create binary mask for this region
    region_mask = labeled_mask == region_label

    # Get the boundary using erosion
    eroded = binary_erosion(region_mask)
    boundary = region_mask ^ eroded  # XOR: region minus eroded = boundary

    # Store boundaries
    boundaries |= boundary  # add to global boundary mask

# Optional: visualize
plt.imshow(mask,cmap='tab10')
plt.imshow(boundaries, cmap='gray',alpha=0.5)
plt.title('Boundaries of Connected Components')
plt.axis('off')
plt.show()
# %%
from scipy.interpolate import splprep, splev
from skimage.measure import find_contours
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

# Label connected components
labeled_mask = label(mask)
# Optionally remove very small regions
min_size = 10  # adjust this threshold
labeled_mask = remove_small_objects(labeled_mask, min_size)
def merge_small_regions(label_img, min_area=10):
    props = regionprops(label_img)
    output = np.copy(label_img)
    small_labels = [r.label for r in props if r.area < min_area]
    for small_lbl in small_labels:
        mask_small = label_img == small_lbl
        # Get border of small region
        dist = distance_transform_edt(~mask_small)
        # Get all other labels
        other_mask = (label_img != 0) & (label_img != small_lbl)
        nearest = label_img[dist == dist[other_mask].min()]
        new_lbl = np.bincount(nearest).argmax()
        output[mask_small] = new_lbl
    
    return output
labeled_mask = merge_small_regions(labeled_mask, min_area=100)
contours = []
for region_label in np.unique(labeled_mask):
    if region_label == 0:
        continue
    binary_region = (labeled_mask == region_label)
    cs = find_contours(binary_region, level=0.5)
    contours.extend(cs)  # Each region may have multiple contours

def smooth_contour(contour, smoothing=0.01, num_points=200):
    x, y = contour[:, 1], contour[:, 0]
    tck, u = splprep([x, y], s=smoothing)
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)
    return np.vstack((y_new, x_new)).T

#%%
s = 50000
smoothed_contours = [smooth_contour(c,smoothing=s) for c in contours]
plt.imshow(mask, cmap='gray')
for sc in smoothed_contours:
    plt.plot(sc[:, 1], sc[:, 0], '-r', linewidth=0.5)
plt.axis('off')
plt.title(f"Smoothed Boundaries,smoothing factor{s}")
plt.show()

# %%
# --- Display in Napari ---
import napari
viewer = napari.Viewer()
viewer.add_labels(labeled_mask, name='Labeled Mask')

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
