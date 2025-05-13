import numpy as np
import napari
import os
from magicgui import magicgui, widgets
from image_reader import wrap_image
import tifffile as tif
from visulized_fibertract import filter_mask_given_acronym_lst, compute_cuboid_roi, compute_annotation
import time
from scipy.ndimage import zoom
from napari_view_utilis import save_remove_layer_given_full_name, toggle_layer_visibility

class SimpleViewer(widgets.Container):
    def __init__(self, viewer1: napari.Viewer, viewer2: napari.Viewer):
        super().__init__()
        self.paras = {
            'img_pth': "/home/confetti/e5_data/t1779/t1779.ims",
            'mask_pth': "/home/confetti/e5_data/t1779/register_data/r32_ims_downsample_561_register/registered_atlas.tiff",
            'mlp_feats_pth':"/home/confetti/e5_data/t1779/t1779.ims",
            'atlas_vs':25,  #for image from resoluton d: vs(img_rd)=2**d
            'mip_thickness':8,
            'SHOW_ANNO':True, #control whether to show region annotation on mask
            'acronym':True, # region annotation type: acronym or region_id
            'show_regions':  None, #['fi','LD'], #set 'None' to not use this term
            'target_region': ['fi'],
            'feats_stride':16,
            'feats_offset':[0,0,0],
            'feats_size':[128,128,128],   
            'opacity':0.28
        }

        self.lr_mask =tif.imread(self.paras['mask_pth'])

        #for viewer1
        self.viewer1= viewer1
        self.roi_layer = viewer1.add_image(np.zeros((32,32,32),dtype =np.uint16),name ='roi')

        #for viewer2
        self.viewer2= viewer2
        self.aux_slice_layer = viewer2.add_image(np.zeros((32,32),dtype = np.uint16),name='aux_slice')
        self.aux_mask_layer = viewer2.add_labels(np.zeros((32,32),dtype = np.uint16),name='aux_mask',opacity =self.paras['opacity'])
        dummpy_polygon = np.array([
            [1,1],
            [1,1],
                                   ])
        self.aux_polygon_layer = viewer2.add_shapes(dummpy_polygon,name='polygon',edge_width=1,edge_color='cyan',opacity =self.paras['opacity'])
        self.aux_acronym_layer = viewer2.add_points(dummpy_polygon,name='region_annotation_id')
        #order is (z,y,x) unit is in the r0
        self.roi_offset = np.array([1536,64,1536])
        self.roi_size = np.array([32,64,128])

        self.register_callbacks()

    def register_callbacks(self):
        self.viewer1.bind_key('f', self.refresh, overwrite=True)
        self.viewer2.bind_key('f', self.refresh, overwrite=True)

        self.img_pth_input = widgets.FileEdit( )
        self.img_pth_submit_button = widgets.PushButton(text='Submit')
        self.img_pth_submit_button.changed.connect(self.on_image_reading)

        self.img_pth_input_row = widgets.Container(
            widgets=[self.img_pth_input, self.img_pth_submit_button],
            layout='horizontal',
            label='image_path'
        )
    




        self.button0 = widgets.PushButton(text="refresh")
        self.button0.clicked.connect(self.refresh)

        self.region_input = widgets.LineEdit(value='STN')
        self.submit_button = widgets.PushButton(text="Submit")
        self.submit_button.changed.connect(self.get_target_region)
        # Group them in one horizontal line
        self.target_region_input_row = widgets.Container(
            widgets=[self.region_input, self.submit_button],
            layout='horizontal',
            label='target_region'
        )


        self.which_half_dropdown = widgets.ComboBox(
            choices=['left','right'],
            label='target half brain',
            tooltip='Select target half brain',
            value = 'left'
        )
        
        # value change of these 6 parameters will not trigger anything, after user change any value of these, click refresh to load new data
        self.x_size = widgets.Slider(label="x size", value=128, min=0, max=10000)
        self.y_size = widgets.Slider(label="y size", value=128, min=0, max=10000)
        self.z_size = widgets.Slider(label="z size", value=128, min=0, max=10000)
        self.x = widgets.LineEdit(label="x offset", value=0)
        self.y = widgets.LineEdit(label="y offset", value=0)
        self.z = widgets.LineEdit(label="z offset", value=0)
        self.offset = widgets.Container(
            widgets=[self.x, self.y,self.z],
            layout='horizontal',
            label='offset',
        )

        self.channel_dropdown = widgets.ComboBox(
            choices=[], 
            label='Channel',
            tooltip='Select channel'
        )
        self.channel_dropdown.changed.connect(self.on_channel_change)


        self.v1_resol_dropdown = widgets.ComboBox(
            choices=[], 
            label='viewer1:Resol Level',
            tooltip='Select resolution level for viewer1'
        )
        self.v1_resol_dropdown.changed.connect(self.on_v1_resol_changed)
         
        self.v2_resol_dropdown = widgets.ComboBox(
            choices=[], 
            label='viewer2:Resol Level',
            tooltip='Select resolution level for viewer1'
        )
        self.v2_resol_dropdown.changed.connect(self.on_v2_resol_changed)

        self.show_anno_dropdown = widgets.ComboBox(
            choices=['all','target_only','None'],
            label='acronym_mode',
        )
        self.show_anno_dropdown.changed.connect(self.on_acronym_changed)

        self.user_hints = widgets.TextEdit(label='Hints')
         

                
        self.extend([
            self.button0,
            self.img_pth_input_row,
            self.target_region_input_row,
            self.which_half_dropdown,
            self.x_size,
            self.y_size,
            self.z_size,
            self.offset,
            self.channel_dropdown,
            self.v1_resol_dropdown,
            self.v2_resol_dropdown,
            self.show_anno_dropdown,
            self.user_hints,
        ])
            

        self.img_pth_input.set_value(self.paras['img_pth'])


    def on_image_reading(self):
        current = time.time()
        self.image = wrap_image(str(self.img_pth_input.value))

        z_offset,y_offset,x_offset,z_size,y_size,x_size = self.image.rois[0]
        self.x.value = x_offset+x_size//2
        self.y.value = y_offset+y_size//2
        self.z.value = z_offset+z_size//2
        self.x_size.max = x_size
        self.y_size.max = y_size
        self.x_size.max = z_size

        resolution_levels = self.image.resolution_levels
        channels = self.image.channels
        self.channel_dropdown.changed.disconnect(self.on_channel_change)
        self.channel_dropdown.choices = channels
        self.channel_dropdown.value = channels[2]
        self.channel_dropdown.changed.connect(self.on_channel_change)

        self.v1_resol_dropdown.changed.disconnect(self.on_v1_resol_changed)
        self.v1_resol_dropdown.choices = resolution_levels
        self.v1_resol_dropdown.value = resolution_levels[0]
        self.v1_resol_dropdown.changed.connect(self.on_v1_resol_changed)

        self.v2_resol_dropdown.changed.disconnect(self.on_v2_resol_changed)
        self.v2_resol_dropdown.choices = resolution_levels
        self.v2_resol_dropdown.value = resolution_levels[3]
        self.v2_resol_dropdown.changed.connect(self.on_v2_resol_changed)

        self.refresh()
        print(f"####loading image fininshed #### time:{time.time()-current:.4f}")
    

    @property
    def get_roi_size(self):
        return [int(self.z_size.value), int(self.y_size.value), int(self.x_size.value)]

    @property
    def get_roi_offset(self):
        return [int(self.z.value), int(self.y.value), int(self.x.value)]


    def refresh(self):
        if self.image is None:
            return

        roi_offset = [int(float(self.z.value)) ,int(float(self.y.value)), int(float(self.x.value))]
        # roi_size = [int(self.z_size.value), int(self.y_size.value), int(self.x_size.value)]
        roi_size = [
                    min(int(self.z_size.value), 64),
                    min(int(self.y_size.value), 1024),
                    min(int(self.x_size.value), 1024)
                ]

        self.add_data_given_offset_size(roi_offset,roi_size)


        info = "\n".join(f"{key}: {value}" for key, value in self.image.info[self.image.resolution_levels.index(self.v1_resol_dropdown.value)].items())
        self.user_hints.value = info


    def get_mask_from_different_scale(self,target_indexs,target_roi_size,scale,return_ori_scale_img =False):
        """
        scale = source_mask_vs / target_vs
        roi is defined via indexs +roi_size
        """
        source_mask = self.lr_mask
        source_indexs=[int(idx/scale)for idx in target_indexs]
        # lr_roi_size=[int(roi/scale)for roi in roi_size]
        source_roi_size = [max(int(roi / scale), 1) for roi in target_roi_size]
        z,y,x=source_indexs
        z_size,y_size,x_size=source_roi_size

        #mask in order (z,y,x)
        lr_mask_roi=source_mask[z:z+z_size,y:y+y_size,x:x+x_size]
        zoom_factors=[t/s for t,s in zip(target_roi_size,source_roi_size)]
        zoomed_mask_roi=zoom(lr_mask_roi,zoom=zoom_factors,order=0)
        zoomed_mask_roi=np.squeeze(zoomed_mask_roi)
        if return_ori_scale_img:
            return zoomed_mask_roi, np.squeeze(lr_mask_roi)
        else:
            return zoomed_mask_roi

    def get_offset_and_ori_size_warp_region_half(self,margin_ratio = 0.10):
        """
        for one acronym at a time
        acronym ->id -->boundary of region +20% as margin 
        --> lr_ori_offset ,lr_ori_shape -[zoom]-> offset, shape in 1um
        """
        
        target_region = self.region_input.value
        #tranfer copy of mask, in this function, will need to set half of the mask to zeroes
        mask = filter_mask_given_acronym_lst([target_region],self.lr_mask.copy(),half= self.which_half_dropdown.value)
        lr_offset,lr_roi = compute_cuboid_roi(mask,margin_ratio= margin_ratio)
        print(f"In low resol, shape of {target_region} is {lr_roi}, offset is {lr_offset}")
        zoom_factor = self.paras['atlas_vs']
        offset = [int(zoom_factor*item) for item in lr_offset]
        roi_size = [int(zoom_factor*item) for item in lr_roi]
        # roi_size = [
        #     min(int(zoom_factor * item), cap)
        #     for item, cap in zip(lr_roi, [128, 1024, 1024])
        # ]
        return offset,roi_size

    #render the roi border at z_slice by draw a 2d cube
    def add_roi_border_in_v2(self, v2_roi_offset,v2_roi_size):
        idx_ = v2_roi_offset
        roi_ = v2_roi_size 
        roi_polygon = np.array( 
            [
                [idx_[1]           , idx_[2]],
                [idx_[1] + roi_[1] , idx_[2]],
                [idx_[1] + roi_[1] , idx_[2] + roi_[2]],
                [idx_[1]           , idx_[2] + roi_[2]],
            ])

        self.aux_polygon_layer.data = roi_polygon 


    def add_data_given_offset_size(self,_roi_offset,_roi_size):

        (self.z.value, self.y.value, self.x.value) = _roi_offset
        (self.z_size.value, self.y_size.value, self.x_size.value) = _roi_size

        self.v1_refresh() 
        self.v2_refresh()
 

    def get_target_region(self):
        # region_name --> offset, roi_size in r0
        #first check does the given region_name is in the atlast_region_dict, if not prompt hints to let user reenter

        roi_offset, roi_size = self.get_offset_and_ori_size_warp_region_half()
        self.add_data_given_offset_size(roi_offset,roi_size)


    def get_str_and_int_from_widget_resol_level(self,str_resol_level:str):
        level_str = str(str_resol_level)
        level_int = self.image.resolution_levels.index(level_str)
        return level_str,level_int

    def on_v1_resol_changed(self):
        self.v1_refresh()
    
    def on_v2_resol_changed(self, ):
        self.v2_refresh()

    def on_channel_change(self, ):
        if isinstance(self.channel_dropdown.value,str):
            self.refresh()
    
    def v1_refresh(self):
        roi_offset = (int(self.z.value), int(self.y.value), int(self.x.value))  
        roi_size = (int(self.z_size.value), int(self.y_size.value), int(self.x_size.value))

        channel_str = str(self.channel_dropdown.value)
        # get corrspond roi in v1_level resl and add it to v1
        #convert offset and size into v1_level
        
        v1_level_str,v1_level_int = self.get_str_and_int_from_widget_resol_level(self.v1_resol_dropdown.value)

        print(f"in v1_refresh: {type(roi_offset[0])}")
        v1_roi_offset = [int(x/(2**v1_level_int)) for x in roi_offset]
        v1_roi_size= [max(int(x/(2**v1_level_int)) ,1)for x in roi_size]
        roi = self.image.from_roi(coords=[*v1_roi_offset, *v1_roi_size],level=v1_level_str,channel=channel_str)
        self.roi_layer.data = roi
        self.roi_layer.contrast_limits = (0,np.percentile(roi,99)+600)

        # self.viewer1.camera.center=(0,v1_roi_size[1],v1_roi_size[2])
    
    def v2_refresh(self):
        begin= time.time()
        roi_offset = (int(self.z.value), int(self.y.value), int(self.x.value))  
        roi_size = (int(self.z_size.value), int(self.y_size.value), int(self.x_size.value))
        self.middle_z_idx=roi_offset[0]+roi_size[0]//2

        channel_str = str(self.channel_dropdown.value)

        v2_level_str,v2_level_int = self.get_str_and_int_from_widget_resol_level(self.v2_resol_dropdown.value)
        v2_roi_offset = [int(x/(2**v2_level_int)) for x in roi_offset]
        v2_roi_size= [max(int(x/(2**v2_level_int)) ,1)for x in roi_size]
        v2_z_idx = v2_roi_offset[0] + int( v2_roi_size[0]//2)

        current = time.time()
        aux_z_slice=self.image.from_slice(v2_z_idx,level=v2_level_str,channel=channel_str,index_pos=0,mip_thick= max(int(self.paras['mip_thickness']//(2**v2_level_int)),1))
        print(f"load 2d slice time :{time.time() -current :.5f}")
        self.aux_slice_layer.data = aux_z_slice

        # compute the percentile will consume a lot of time, for a 2d slice ~(12000,13000), will consume 14 second
        # self.aux_slice_layer.contrast_limits = (0,np.percentile(aux_z_slice,99)+600)
        self.aux_slice_layer.contrast_limits = (0,3000)


        #add slice_mask
        v2_slice_shape=self.image.rois[v2_level_int][-2:]
        slice_mask ,lr_slice_mask= self.get_mask_from_different_scale(
                                                                target_indexs=[v2_z_idx,0,0],
                                                                target_roi_size=[1,*v2_slice_shape],
                                                                scale= self.paras['atlas_vs']/ (2** v2_level_int),
                                                                return_ori_scale_img=True,
                                                                )
        self.lr_slice_mask = lr_slice_mask                                                   
        self.aux_mask_layer.data = slice_mask

        #add roi_border
        self.add_roi_border_in_v2(v2_roi_offset,v2_roi_size)

        #add acronym
        self.on_acronym_changed()
        print(f"finished v2 refresh, time consumed: {time.time() - begin}")
            #adjust camera in sub_viewer
        self.viewer2.camera.zoom=2
        self.viewer2.camera.center=(0,v2_roi_offset[1],v2_roi_offset[2])







    def on_acronym_changed(self):
        acronym_mode = self.show_anno_dropdown.value

        if acronym_mode == 'None':
            save_remove_layer_given_full_name(self.viewer2.layers,'region_annotation_id')
            return

        else:
            save_remove_layer_given_full_name(self.viewer2.layers,'region_annotation_id')
            if acronym_mode =='all':
                target_mask = self.lr_slice_mask 
            else:
                target_region = self.region_input.value
                #tranfer copy of mask, in this function, will need to set half of the mask to zeroes
                target_mask = filter_mask_given_acronym_lst([target_region],self.lr_slice_mask.copy(),half= self.which_half_dropdown.value)

            region_centers, region_annotations = compute_annotation(target_mask,acronym=True)
            region_centers = region_centers.astype(float)

            v2_level_str,v2_level_int = self.get_str_and_int_from_widget_resol_level(self.v2_resol_dropdown.value)
            region_centers *= self.paras['atlas_vs']/ (2**v2_level_int)

            self.viewer2.add_points(
                region_centers,
                properties={"label": region_annotations},
                text={
                    "string": "{label}",
                    "anchor": "center",
                    "color": "orange",
                    "size": 8,
                },
                size=4,
                face_color="transparent",
                border_color= 'transparent',
                name="region_annotation_id",
            )
            return
            


        


        

