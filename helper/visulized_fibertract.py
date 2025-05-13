import tifffile as tif
import napari
import numpy as np
from typing import List
from scipy.ndimage import center_of_mass
import time

from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
atlas = BrainGlobeAtlas("allen_mouse_25um")

def get_binary_mask(parent_acronym,mask,bg_atlas):
    #change all the descendent id belonogs to target_id to target_id
    #change into binary mask
    """
    Convert all descendants of a given structure in the mask volume to the ID of the parent structure.

    Parameters:
        mask (numpy.ndarray): The mask volume to modify.
        bg_atlas: The BrainGlobe Atlas instance.
        parent_acronym (str): The acronym of the parent structure (e.g., "fiber tracts").
    """
    
    
    descendant_acronyms = bg_atlas.get_structure_descendants(parent_acronym)
    parent_id = bg_atlas.structures[parent_acronym]['id']
    
    # Get IDs for all descendants
    descendant_ids = [bg_atlas.structures[desc]["id"] for desc in descendant_acronyms]
    
    # Create a copy of the mask to modify
    modified_mask = np.zeros_like(mask)
    
    # Replace all descendant IDs in the mask with the parent ID
    for desc_id in descendant_ids:
        modified_mask[mask == desc_id] = 1 

    #for parent_acronym
    modified_mask[mask == parent_id] = 1
    
    
    return modified_mask

def acronym2id(acronymn, atlas)->int:
    id =int(atlas.structures[acronymn]['id'])
    return id

def id2acronym(id: str | int, atlas) -> str:
    id = str(id)
    return atlas.structures[id]['acronym']


def filter_mask_given_acronym_lst(acronym_lst: List[str], mask:np.ndarray, half: str = 'left'):
    """
    mask: 2d or 3d
    """
    if mask.ndim ==2:
        mask = mask[None,:]
        print(f"now mask dim is {mask.ndim}")
    if half in ['left', 'right']:
        half_x = mask.shape[-1] // 2
        if half == 'left':
            mask[:, :, half_x:] = 0  # Keep left, zero right
        elif half == 'right':
            mask[:, :, :half_x] = 0  # Keep right, zero left
    else:
        print(f"not choose which half, handle at whole brain") 

    if acronym_lst == None:
        return mask
    atlas = BrainGlobeAtlas("allen_mouse_25um")
    modified_mask = np.zeros_like(mask)
    for idx, acronym in enumerate(acronym_lst):
        binary_mask = get_binary_mask(acronym,mask,atlas)
        modified_mask[binary_mask == 1] = acronym2id(acronym,atlas) 
    return np.squeeze(modified_mask)

def compute_cuboid_roi(mask, margin_ratio=0.2):
    """
    Compute the offset and ROI size of a cuboid that wraps a 3D binary mask with a margin.
    
    Parameters:
        mask (numpy.ndarray): A binary 3D mask of shape (Z, Y, X).
        margin_ratio (float): The margin ratio relative to the ROI size in each dimension.
        
    Returns:
        offset (tuple): The starting coordinate (z_offset, y_offset, x_offset) of the cuboid.
        roi_size (tuple): The size (z_size, y_size, x_size) of the cuboid.
    Example:
        >>> import numpy as np
        >>> mask = np.zeros((100, 100, 100), dtype=np.uint8)
        >>> mask[30:70, 40:80, 50:90] = 1  # Example cuboid mask
        >>> offset, roi_size = compute_cuboid_roi(mask)
        >>> offset
        (24, 34, 44)
        >>> roi_size
        (52, 52, 52)
    """
    # Get the coordinates of non-zero elements in the mask
    coords = np.argwhere(mask > 0)
    
    # Calculate the bounding box of the mask
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0) + 1  # Add 1 to include the boundary
    
    # Compute the size of the bounding box
    bbox_size = max_coords - min_coords
    
    # Calculate the margin in each direction
    margin = np.ceil(margin_ratio * bbox_size).astype(int)
    
    # Compute the offset and size of the ROI with margin
    offset = np.maximum(min_coords - margin, 0)  # Ensure non-negative offset
    roi_size = bbox_size + 2 * margin  # Add margin on both sides
    
    return tuple(offset), tuple(roi_size)




def find_inside_centroid(mask, label_value)->np.ndarray:
    # Extract the region for the specific label
    region = (mask == label_value)
    
    # Compute the initial center of mass
    candidate_center = center_of_mass(region)
    
    # Find all coordinates within the region
    region_coords = np.argwhere(region)
    
    # Compute the Euclidean distance from the candidate to all region points
    distances = np.linalg.norm(region_coords - candidate_center, axis=1)
    
    # Find the region coordinate closest to the candidate center
    closest_idx = np.argmin(distances)
    return region_coords[closest_idx]

def _compute_annotation(mask,acronym = False,verbose = True):


    current = time.time()
    label_indices = np.unique(mask)
    if len(label_indices) <= 1:  # only background
        centers = np.empty((0, 2))  # or whatever shape your centroids normally have
    else:
        label_indices = label_indices[1:]# skip background(label 0 )
        centers = np.array([find_inside_centroid(mask, label) for label in label_indices])

    if acronym:
        text_annotations_lst = [f"{atlas.structures[label]['acronym']}"for label in label_indices]
    else:
        text_annotations_lst = [f"{int(label)}"for label in label_indices]

    if verbose:
        print(f"compute annotation : {time.time() - current}")


    return centers,text_annotations_lst

def compute_annotation(mask,acronym=False,verbose = False):
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
    middle_width = int(mask.shape[-1])//2
    l_centers, l_annotations = _compute_annotation(mask[:,:middle_width],acronym,verbose)
    r_centers, r_annotations = _compute_annotation(mask[:,middle_width:],acronym,verbose)

    r_centers[:,1] += middle_width #r_centers need a horizontal shift
    
    centers = np.concatenate((l_centers,r_centers),axis=0 )
    annotations = l_annotations + r_annotations
    return centers, annotations



if __name__ == '__main__':
    atlas = BrainGlobeAtlas("allen_mouse_25um")
    # print(atlas.structures[603])
    # print(atlas.get_structure_descendants("fiber tracts"))


    mask = tif.imread("/home/confetti/mnt/data/processed/t1779/r32_ims_downsample_561_register/registered_atlas.tiff")
    raw = tif.imread('/home/confetti/mnt/data/processed/t1779/r32_ims_downsample_561_register/downsampled.tiff')
    ft_mask = filter_mask_given_acronym_lst(['fi'],mask)

    #TODO check the radius of most fiber_tract region
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_labels(ft_mask,opacity=0.27)
    viewer.add_image(raw)
    napari.run()

