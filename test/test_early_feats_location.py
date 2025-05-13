#%%
import numpy as np
import napari
import zarr
from image_reader import Ims_Image
level = 0
stride = 16
ims_vol = Ims_Image(ims_path="/home/confetti/e5_data/t1779/t1779.ims", channel=2)
raw_volume_size =ims_vol.rois[level][3:] #the data shape at r3 for test
print(f"raw_volume_size{raw_volume_size}")
#%%
whole_volume_size = [int(element//2) for element in raw_volume_size]
whole_volume_offset = [int(element//4) for element in raw_volume_size]
# valid_offset = [ int(x + int((3/2) *stride)) for x in whole_volume_offset]
# valid_size = [ int(x - int((3/2) *stride)) for x in whole_volume_size]
# lb = valid_offset
# hb = [ x+ y for x, y in zip(valid_offset,valid_size)] 
# roi_offset is 3d
roi_offset = [whole_volume_offset[0]+32,whole_volume_offset[1],whole_volume_offset[2]] 
roi_size =[1,whole_volume_size[1],whole_volume_size[2]]
raw_img = ims_vol.from_roi(coords=[*roi_offset,*roi_size],level=0)
print(f"reading from ims, raw_img shape {raw_img.shape}")
#%%
save_path= '/home/confetti/data/t1779/mlp_feats.zarr'
z_arr = zarr.open_array(save_path)
C,D,H,W = z_arr.shape
print(z_arr.shape)

def crop_3d_slice(volume, dim, index,reshape2lst =True):
    C = volume.shape[0]
    # Perform slicing
    if dim == 1:       # XY slice at given Z index
        slice_2d = volume[:,index, :, :]
    elif dim == 2:     # XZ slice at given Y index
        slice_2d = volume[:,:, index, :]
    elif dim == 3:     # YZ slice at given X index
        slice_2d = volume[:,:, :, index]
    if reshape2lst:
        feats_lst = slice_2d.reshape(C,-1).T
        return feats_lst
    else:
        return slice_2d

def cos_theta_plot(idx, encoded, img_shape):
    att = encoded @ encoded[idx]
    img = att.reshape(img_shape) 
    return img


#%%
z_feats_0 = crop_3d_slice(z_arr,1,000)

ncc_feats = cos_theta_plot(idx=60000,encoded=z_feats_0,img_shape=(H,W))
zoomed_ncc_feats = np.kron(ncc_feats,np.ones((stride,stride)))
padded = np.pad(zoomed_ncc_feats,pad_width=((24,24),(24,24)),constant_values=0)

viewer = napari.Viewer()
viewer.add_image(raw_img,name ='roi')
viewer.add_image(zoomed_ncc_feats,opacity=0.7,name='feats')
viewer.add_image(padded,opacity=0.7,name='padd_feats')
napari.run()

# %%
print(f"zoomed_ncc_shape:{zoomed_ncc_feats.shape}")
print(f"raw_shape {raw_img.shape}")

# %%
