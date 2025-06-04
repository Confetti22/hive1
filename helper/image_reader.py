import numpy as np
import h5py
import zarr
import re
from tifffile import imread
import random

class Ims_Image():
    '''
    the old ims_reader
    ims image: [z,y,x]
    input roi and returned image: [z,y,x]
    '''
    def __init__(self,ims_path,channel=0):
        self.hdf = h5py.File(ims_path,'r')
        level_keys = list(self.hdf['DataSet'].keys())
        channel_keys = [key for key in self.hdf['DataSet'][level_keys[0]]['TimePoint 0']]
        self.images = [self.hdf['DataSet'][key]['TimePoint 0'][channel_keys[channel]]['Data'] for key in level_keys]
        # image_info = self.hdf.get('DataSetInfo')['Image'].attrs
        # print(eval(image_info['ExtMax3']))
        self.rois = []
        self.info = self.get_info()
        for i in self.info:
            # assure rois in order (z,y,x)
            self.rois.append(i['origin'] + i['data_shape'])
        self.roi = self.rois[0]


    def __getitem__(self, indices, level=0):
        z_min, z_max = indices[0].start, indices[0].stop
        y_min, y_max = indices[1].start, indices[1].stop
        x_min, x_max = indices[2].start, indices[2].stop
        z_slice = slice(z_min-self.rois[level][0],z_max-self.rois[level][0])
        y_slice = slice(y_min-self.rois[level][1],y_max-self.rois[level][1])
        x_slice = slice(x_min-self.rois[level][2],x_max-self.rois[level][2])
        return self.images[level][z_slice,y_slice,x_slice]


    def from_roi(self, coords, level=0):
        # coords: [z_offset,y_offset,x_offset,z_size,y_size,x_size]
        # wanted coords
        z_min, z_max = coords[0], coords[3]+coords[0]
        y_min, y_max = coords[1], coords[4]+coords[1]
        x_min, x_max = coords[2], coords[5]+coords[2]

        # add padding
        # bounded coords
        [zlb,ylb,xlb] = self.rois[level][0:3] 
        [zhb,yhb,xhb] = [i+j for i,j in zip(self.rois[level][:3],self.rois[level][3:])]
        zlp = max(zlb-z_min,0)
        zhp = max(z_max-zhb,0)
        ylp = max(ylb-y_min,0)
        yhp = max(y_max-yhb,0)
        xlp = max(xlb-x_min,0)
        xhp = max(x_max-xhb,0)

        z_slice = slice(z_min-self.rois[level][0]+zlp,z_max-self.rois[level][0]-zhp) 
        y_slice = slice(y_min-self.rois[level][1]+ylp,y_max-self.rois[level][1]-yhp)
        x_slice = slice(x_min-self.rois[level][2]+xlp,x_max-self.rois[level][2]-xhp)
        img = self.images[level][z_slice,y_slice,x_slice]

        padded = np.pad(img, ((zlp, zhp), (ylp, yhp), (xlp, xhp)), 'constant')

        return padded


    def from_slice(self,index,level,index_pos=0,mip_thick = 1):
        """
        index_pos: 0 for z_slice, 1 for y_slice, 2 for x_slice
        mip_thick ==1 will only cut a plane
        for mip_thick > 1 , will acquire mip 
        """
        lb = self.rois[level][0:3] 
        hb = [i+j for i,j in zip(self.rois[level][:3],self.rois[level][3:])]
        assert lb[index_pos] <= index < hb[index_pos], \
            f"Index {index} out of range for axis {index_pos}. Must be between {lb[index_pos]} and {hb[index_pos] - 1}."
        
        half_thick = int(mip_thick//2)
        if half_thick == 0: #for mip_thick == 1, which is just a cut plane of one pixel thickness
            l_idx = index
            r_idx = index +1
        else: # for mip_thick > 1, which will apply mip to acquire one pixel plane
            l_idx = index - half_thick
            r_idx = index + half_thick
            if mip_thick % 2 ==1: # for odd mip_thickness
                r_idx += 1
            
        # Slicing the 3D array based on the given index and axis
        if index_pos == 0:  # Slice along the z-axis (extract a z_slice)
            slice_2d = self.images[level][l_idx:r_idx, lb[1]:hb[1], lb[2]:hb[2]]
            slice_2d = np.max(slice_2d,axis=0)
        elif index_pos == 1:  # Slice along the y-axis (extract a y_slice)
            slice_2d = self.images[level][lb[0]:hb[0], l_idx:r_idx, lb[2]:hb[2]]
            slice_2d = np.max(slice_2d,axis=1)
        elif index_pos == 2:  # Slice along the x-axis (extract an x_slice)
            slice_2d = self.images[level][lb[0]:hb[0], lb[1]:hb[1],  l_idx:r_idx]
            slice_2d = np.max(slice_2d,axis=2)
        else:
            raise ValueError(f"Invalid index_pos {index_pos}. Must be 0 (z), 1 (y), or 2 (x).")
        # Remove any extra dimensions
        slice_2d = np.squeeze(slice_2d)
        
        return slice_2d
        






    def from_local(self, coords, level=0):
        # coords: [z_offset,y_offset,x_offset,z_size,y_size,x_size]
        z_min, z_max = coords[0], coords[3]+coords[0]
        y_min, y_max = coords[1], coords[4]+coords[1]
        x_min, x_max = coords[2], coords[5]+coords[2]

        z_slice = slice(z_min,z_max) 
        y_slice = slice(y_min,y_max)
        x_slice = slice(x_min,x_max)
        return self.images[level][z_slice,y_slice,x_slice]


    def get_info(self):
        if 'DataSetInfo' in self.hdf.keys():
            image_info = self.hdf.get('DataSetInfo')['Image'].attrs
            # calculate physical size
            extents = []
            for k in ['ExtMin2', 'ExtMin1', 'ExtMin0', 'ExtMax2', 'ExtMax1', 'ExtMax0']:
                extents.append(eval(image_info[k]))
            dims_physical = []
            for i in range(3):
                dims_physical.append(extents[3+i]-extents[i])
            origin = [int(extents[0]), int(extents[1]), int(extents[2])]
        else:
            origin = [0,0,0]
            dims_physical = None

        info = []
        # get data size
        level_keys = list(self.hdf['DataSet'].keys())
        for i, level in enumerate(level_keys):
            hdata_group = self.hdf['DataSet'][level]['TimePoint 0']['Channel 0']
            data = hdata_group['Data']
            dims_data = []
            for k in ["ImageSizeZ", "ImageSizeY", "ImageSizeX"]:
                dims_data.append(int(eval(hdata_group.attrs.get(k))))
            if dims_physical == None:
                dims_physical = dims_data
            spacing = [dims_physical[0]/dims_data[0], dims_physical[1]/dims_data[1], dims_physical[2]/dims_data[2]]
            info.append(
                {
                    'level':level,
                    'dims_physical':dims_physical,
                    'image_size':dims_data,
                    'data_shape':[data.shape[0],data.shape[1],data.shape[2]],
                    'data_chunks':data.chunks,
                    'spacing':spacing,
                    'origin':origin
                }
            )
        return info
    
    def get_random_roi(self,
                    filter=lambda x:np.mean(x)>=150,
                    roi_size=(64,64,64),
                    level=0,
                    skip_gap = False,
                    sample_range = None,
                    margin = 128, #sampled roi will be within the interior of index 
                    ):

        """
        random sample a roi of size (z_extend,y_extend,x_extend) that pass the filter check
        return the start indexes
        """
        foreground_sample_flag=False
        #shape: (z,y,x)
        info=self.get_info()
        sample_lb = [0,0,0]
        sample_rb=info[level]['data_shape']
        if sample_range:
            sample_lb = [idx_range[0] for idx_range in sample_range]
            sample_rb = [ idx_range[1] for idx_range in sample_range]

        if skip_gap:
            #for skipping the gap between slices
            start = 5
            step = 300
            end = step - start - roi_size[0] 
            limit = sample_rb[0]-roi_size[0] - margin 
            intervals = []
            
            #add one step to skip the fist z slice
            current_start = start + sample_lb[0]
            current_end = end + sample_lb[0]
        
            # Generate intervals
            while current_end <= limit:
                intervals.append((current_start, current_end))
                current_start += step
                current_end += step
        

        while not foreground_sample_flag:

            if skip_gap:
                chosen_interval = random.choice(intervals)
                z_idx = random.randint(chosen_interval[0],chosen_interval[1])
            else:
                z_idx=np.random.randint(sample_lb[0] + margin ,sample_rb[0]-roi_size[0] - margin) 
            y_idx=np.random.randint(sample_lb[1] + margin ,sample_rb[1]-roi_size[1] - margin) 
            x_idx=np.random.randint(sample_lb[2] + margin ,sample_rb[2]-roi_size[2] - margin) 
            roi=self.from_roi(coords=[z_idx,y_idx,x_idx,roi_size[0],roi_size[1],roi_size[2]],level=level)
            roi=roi.reshape(roi_size[0],roi_size[1],roi_size[2])
            roi=np.squeeze(roi)
            
            #filter check
            foreground_sample_flag=filter(roi)

        return roi, [z_idx,y_idx,x_idx]
    
    def sample_within_range(
                        self,
                        center,
                        radius,
                        filter=lambda x:np.mean(x)>=150,
                        roi_size=(64,64,64),
                        level=0,
                        skip_gap = False,
                        margin = 128, #sampled roi will be within the interior of index 
                        ):
        foreground_sample_flag=False

        idx_range = [ [idx - radius, idx + radius] for idx in center]

        while not foreground_sample_flag:

            z_idx = np.random.randint(idx_range[0][0],idx_range[0][1])
            y_idx = np.random.randint(idx_range[1][0],idx_range[1][1])
            x_idx = np.random.randint(idx_range[2][0],idx_range[2][1])
        
            roi=self.from_roi(coords=[z_idx,y_idx,x_idx,roi_size[0],roi_size[1],roi_size[2]],level=level)
            roi=roi.reshape(roi_size[0],roi_size[1],roi_size[2])
            roi=np.squeeze(roi)
            
            #filter check
            foreground_sample_flag=filter(roi)
        
        sampled_idx = np.array([z_idx,y_idx,x_idx])
        l2_dist = np.linalg.norm(sampled_idx - center)
        print(f"sampled distance:{l2_dist}")
        return roi , l2_dist



class _Ims():
    '''
    the new ims reader
    '''
    def __init__(self,ims_path):
        self.hdf = h5py.File(ims_path,'r')
        self.rois = []
        self.info = self.get_info()
        for i in self.info:
            self.rois.append([int(j/k) for j,k in zip(i['origin'],i['spacing'])] + i['image_size'])
        self.extension = self.info[0]['origin'] + self.info[0]['dims_physical']
        self.dataset = self.hdf.get('DataSet')
        self.time_point_key = 'TimePoint 0'
        self.resolution_levels = list(self.dataset.keys())
        self.channels = self.list_channels(self.resolution_levels[0])


    def list_channels(self, resolution_level):
        res_level = self.dataset.get(resolution_level)
        if res_level is None:
            raise KeyError(f"Resolution level '{resolution_level}' not found.")
        
        time_point_group = res_level.get(self.time_point_key)
        if time_point_group is None:
            raise KeyError(f"Time point '{self.time_point_key}' not found in resolution level '{resolution_level}'.")
        
        return list(time_point_group.keys())


    def __getitem__(self, indices, level=0, channel=0):
        z_min, z_max = indices[0].start, indices[0].stop
        y_min, y_max = indices[1].start, indices[1].stop
        x_min, x_max = indices[2].start, indices[2].stop
        z_slice = slice(z_min-self.rois[level][0],z_max-self.rois[level][0])
        y_slice = slice(y_min-self.rois[level][1],y_max-self.rois[level][1])
        x_slice = slice(x_min-self.rois[level][2],x_max-self.rois[level][2])
        return self.images[level][z_slice,y_slice,x_slice]



    def from_roi(self, coords, level:str|int, channel:str|int, padding='constant'):
        # coords: [x_offset,y_offset,z_offset,x_size,y_size,z_size]
        if isinstance(level,str):
            level = self.resolution_levels.index(level)
        if isinstance(channel,str):
            channel = self.channels.index(channel)

        # coords: [z_offset,y_offset,x_offset,z_size,y_size,x_size]
        # wanted coords
        z_min, z_max = coords[0], coords[3]+coords[0]
        y_min, y_max = coords[1], coords[4]+coords[1]
        x_min, x_max = coords[2], coords[5]+coords[2]

        # add padding
        # bounded coords
        [zlb,ylb,xlb] = self.rois[level][0:3] 
        [zhb,yhb,xhb] = [i+j for i,j in zip(self.rois[level][:3],self.rois[level][3:])]
        zlp = max(zlb-z_min,0)
        zhp = max(z_max-zhb,0)
        ylp = max(ylb-y_min,0)
        yhp = max(y_max-yhb,0)
        xlp = max(xlb-x_min,0)
        xhp = max(x_max-xhb,0)

        z_slice = slice(z_min-self.rois[level][0]+zlp,z_max-self.rois[level][0]-zhp) 
        y_slice = slice(y_min-self.rois[level][1]+ylp,y_max-self.rois[level][1]-yhp)
        x_slice = slice(x_min-self.rois[level][2]+xlp,x_max-self.rois[level][2]-xhp)

        if isinstance(level, int) and isinstance(channel, int):
            image = self.dataset[self.resolution_levels[level]][self.time_point_key][self.channels[channel]]['Data']
        else:
            return
        img = image[z_slice,y_slice,x_slice]
        padded = np.pad(img, ((zlp, zhp), (ylp, yhp), (xlp, xhp)), padding)

        return padded

    def from_slice(self,index,level:str, channel:str, index_pos=0,mip_thick = 1):
        """
        index_pos: 0 for z_slice, 1 for y_slice, 2 for x_slice
        mip_thick ==1 will only cut a plane
        for mip_thick > 1 , will acquire mip 
        """
        if isinstance(level,str):
            level = self.resolution_levels.index(level)
        if isinstance(channel,str):
            channel = self.channels.index(channel)

        lb = self.rois[level][0:3] 
        hb = [i+j for i,j in zip(self.rois[level][:3],self.rois[level][3:])]
        assert lb[index_pos] <= index < hb[index_pos], \
            f"Index {index} out of range for axis {index_pos}. Must be between {lb[index_pos]} and {hb[index_pos] - 1}."
        
        half_thick = int(mip_thick//2)
        if half_thick == 0: #for mip_thick == 1, which is just a cut plane of one pixel thickness
            l_idx = index
            r_idx = index +1
        else: # for mip_thick > 1, which will apply mip to acquire one pixel plane
            l_idx = index - half_thick
            r_idx = index + half_thick
            if mip_thick % 2 ==1: # for odd mip_thickness
                r_idx += 1
            
        image = self.dataset[self.resolution_levels[level]][self.time_point_key][self.channels[channel]]['Data']

        if index_pos == 0:  # Slice along the z-axis
            z_slice = slice(l_idx, r_idx)
            y_slice = slice(lb[1], hb[1])
            x_slice = slice(lb[2], hb[2])
            sub_image = image[z_slice, y_slice, x_slice]
            slice_2d = np.max(sub_image, axis=0)

        elif index_pos == 1:  # Slice along the y-axis
            z_slice = slice(lb[0], hb[0])
            y_slice = slice(l_idx, r_idx)
            x_slice = slice(lb[2], hb[2])
            sub_image = image[z_slice, y_slice, x_slice]
            slice_2d = np.max(sub_image, axis=1)

        elif index_pos == 2:  # Slice along the x-axis
            z_slice = slice(lb[0], hb[0])
            y_slice = slice(lb[1], hb[1])
            x_slice = slice(l_idx, r_idx)
            sub_image = image[z_slice, y_slice, x_slice]
            slice_2d = np.max(sub_image, axis=2)
        else:
            raise ValueError(f"Invalid index_pos {index_pos}. Must be 0 (z), 1 (y), or 2 (x).")
        # Remove any extra dimensions
        slice_2d = np.squeeze(slice_2d)
        return slice_2d
        
    

    def get_info(self):
        if 'DataSetInfo' in self.hdf.keys():
            image_info = self.hdf.get('DataSetInfo')['Image'].attrs
            # calculate physical size
            extents = []
            for k in ['ExtMin0', 'ExtMin1', 'ExtMin2', 'ExtMax0', 'ExtMax1', 'ExtMax2']:
                extents.append(eval(image_info[k]))
            dims_physical = []
            for i in range(3):
                dims_physical.append(extents[3+i]-extents[i])
            origin = [int(extents[0]), int(extents[1]), int(extents[2])]
        else:
            origin = [0,0,0]
            dims_physical = None

        info = []
        # get data size
        level_keys = list(self.hdf['DataSet'].keys())
        for i, level in enumerate(level_keys):
            hdata_group = self.hdf['DataSet'][level]['TimePoint 0']['Channel 0']
            data = hdata_group['Data']
            dims_data = []
            for k in ["ImageSizeX", "ImageSizeY", "ImageSizeZ"]:
                dims_data.append(int(eval(hdata_group.attrs.get(k))))
            if dims_physical == None:
                dims_physical = dims_data
            spacing = [dims_physical[0]/dims_data[0], dims_physical[1]/dims_data[1], dims_physical[2]/dims_data[2]]
            info.append(
                {
                    'level':level,
                    'dims_physical':dims_physical,
                    'image_size':dims_data,
                    'data_shape':[data.shape[2],data.shape[1],data.shape[0]],
                    'data_chunks':[data.chunks[2],data.chunks[1],data.chunks[0]],
                    'spacing':spacing,
                    'origin':origin
                }
            )
        return info



class ZipZarr():
    '''
    Load hierachical image data of several resolution levels like:
        ├── 1um uint16
        ├── 2um uint16
        ├── 4um uint16
        ├── 8um uint16
        └── 16um uint16
    '''
    def __init__(self,image_path):
        self.store = zarr.open(image_path,mode='r')
        if 'nm' in list(self.store.keys())[0]:
            self.store = self.store['488nm_10X']
        resolution_dict = {
            '1um': [],
            '2um': [],
            '4um': [],
            '8um': [],
            '16um': []
        }
        resolutions = [int(re.findall(r'\d+', dataset_name)[0]) for dataset_name in resolution_dict.keys()]
        for dataset in self.store:
            if '1um' in dataset:
                resolution_dict['1um'] = dataset
            elif '2um' in dataset:
                resolution_dict['2um'] = dataset
            elif '4um' in dataset:
                resolution_dict['4um'] = dataset
            elif '8um' in dataset:
                resolution_dict['8um'] = dataset
            elif '16um' in dataset:
                resolution_dict['16um'] = dataset
        self.images = [self.store[dataset] for key, dataset in resolution_dict.items() if dataset != []]
        self.roi = [0,0,0] + list(self.images[0].shape)
        self.info = []
        self.rois = []
        for i,image in enumerate(self.images):
            self.info.append(
                {
                    'level': i,
                    'spacing': [resolutions[i],resolutions[i],resolutions[i]],
                    'image_size': list(image.shape)
                }
            )
            self.rois.append([0,0,0]+list(image.shape))


    def __getitem__(self, indices, level=0):
        x_min, x_max = indices[0].start, indices[0].stop
        y_min, y_max = indices[1].start, indices[1].stop
        z_min, z_max = indices[2].start, indices[2].stop
        x_slice = slice(x_min-self.rois[level][0],x_max-self.rois[level][0])
        y_slice = slice(y_min-self.rois[level][1],y_max-self.rois[level][1])
        z_slice = slice(z_min-self.rois[level][2],z_max-self.rois[level][2])
        return np.transpose(self.images[level][z_slice,y_slice,x_slice],(2,1,0))


    def from_roi(self, coords, level=0, channel=0,padding='constant'):
        # coords: [x_offset,y_offset,z_offset,x_size,y_size,z_size]
        coords = [int(coord) for coord in coords]
        x_min, x_max = coords[0], coords[3]+coords[0]
        y_min, y_max = coords[1], coords[4]+coords[1]
        z_min, z_max = coords[2], coords[5]+coords[2]
        # add padding
        [xlb,ylb,zlb] = self.rois[level][0:3] 
        [xhb,yhb,zhb] = [i+j for i,j in zip(self.rois[level][:3],self.rois[level][3:])]
        xlp = max(xlb-x_min,0)
        xhp = max(x_max-xhb,0)
        ylp = max(ylb-y_min,0)
        yhp = max(y_max-yhb,0)
        zlp = max(zlb-z_min,0)
        zhp = max(z_max-zhb,0)

        x_slice = slice(x_min-self.rois[level][0]+xlp,x_max-self.rois[level][0]-xhp)
        y_slice = slice(y_min-self.rois[level][1]+ylp,y_max-self.rois[level][1]-yhp)
        z_slice = slice(z_min-self.rois[level][2]+zlp,z_max-self.rois[level][2]-zhp) 
        img = np.transpose(self.images[level][z_slice,y_slice,x_slice],(2,1,0))

        padded = np.pad(img, ((xlp, xhp), (ylp, yhp), (zlp, zhp)), padding)

        return padded


    def get_info(self):
        if 'DataSetInfo' in self.hdf.keys():
            image_info = self.hdf.get('DataSetInfo')['Image'].attrs
            # calculate physical size
            extents = []
            for k in ['ExtMin0', 'ExtMin1', 'ExtMin2', 'ExtMax0', 'ExtMax1', 'ExtMax2']:
                extents.append(eval(image_info[k]))
            dims_physical = []
            for i in range(3):
                dims_physical.append(extents[3+i]-extents[i])
            origin = [int(extents[0]), int(extents[1]), int(extents[2])]
        else:
            origin = [0,0,0]
            dims_physical = None

        info = []
        # get data size
        level_keys = list(self.hdf['DataSet'].keys())
        for i, level in enumerate(level_keys):
            hdata_group = self.hdf['DataSet'][level]['TimePoint 0']['Channel 0']
            data = hdata_group['Data']
            dims_data = []
            for k in ["ImageSizeX", "ImageSizeY", "ImageSizeZ"]:
                dims_data.append(int(eval(hdata_group.attrs.get(k))))
            if dims_physical == None:
                dims_physical = dims_data
            spacing = [dims_physical[0]/dims_data[0], dims_physical[1]/dims_data[1], dims_physical[2]/dims_data[2]]
            info.append(
                {
                    'level':level,
                    'dims_physical':dims_physical,
                    'image_size':dims_data,
                    'data_shape':[data.shape[2],data.shape[1],data.shape[0]],
                    'data_chunks':[data.chunks[2],data.chunks[1],data.chunks[0]],
                    'spacing':spacing,
                    'origin':origin
                }
            )
        return info



class Tiff():
    '''
    zarr.attrs['roi'] = [x_offset,y_offset,z_offset,x_size,y_size,z_size]
    To load image directly from global coordinates, wrap .zarr object in this class.
    '''
    def __init__(self,tiff_path):
        self.image = np.squeeze(imread(tiff_path))
        self.roi = [0,0,0] + list(self.image.shape)
        self.rois = [self.roi]
        self.shape = self.roi[3:6]
    
    def __getitem__(self, indices):
        coords = [int(coord) for coord in coords]
        x_min, x_max = indices[0].start, indices[0].stop
        y_min, y_max = indices[1].start, indices[1].stop
        z_min, z_max = indices[2].start, indices[2].stop
        x_slice = slice(x_min-self.roi[0],x_max-self.roi[0])
        y_slice = slice(y_min-self.roi[1],y_max-self.roi[1])
        z_slice = slice(z_min-self.roi[2],z_max-self.roi[2])
        return self.image[x_slice,y_slice,z_slice]

    def from_roi(self, coords, level=0, channel=0, padding='constant'):
        # coords: [x_offset,y_offset,z_offset,x_size,y_size,z_size]
        coords = [int(coord) for coord in coords]
        x_min, x_max = coords[0], coords[3]+coords[0]
        y_min, y_max = coords[1], coords[4]+coords[1]
        z_min, z_max = coords[2], coords[5]+coords[2]
        # add padding
        [xlb,ylb,zlb] = self.roi[0:3] 
        [xhb,yhb,zhb] = [i+j for i,j in zip(self.roi[:3],self.roi[3:])]

        xlp = max(xlb-x_min,0)
        xhp = max(x_max-xhb,0)
        ylp = max(ylb-y_min,0)
        yhp = max(y_max-yhb,0)
        zlp = max(zlb-z_min,0)
        zhp = max(z_max-zhb,0)

        x_slice = slice(x_min-self.roi[0]+xlp,x_max-self.roi[0]-xhp)
        y_slice = slice(y_min-self.roi[1]+ylp,y_max-self.roi[1]-yhp)
        z_slice = slice(z_min-self.roi[2]+zlp,z_max-self.roi[2]-zhp) 
        img = self.image[x_slice,y_slice,z_slice]

        padded = np.pad(img, ((xlp, xhp), (ylp, yhp), (zlp, zhp)), padding) # padding can be constant or reflect

        return padded


def wrap_image(image_path):
    if 'ims' in image_path:
        return _Ims(image_path)
    elif 'zarr.zip' in image_path:
        return ZipZarr(image_path)
    elif 'tif' in image_path:
        return Tiff(image_path)
    else:
        raise Exception("image type not supported yet") 


if __name__ == '__main__':
    pass