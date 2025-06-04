from magicgui import magicgui,widgets
import gc
import napari
import time
import pickle
import tifffile as tif
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from confettii.feat_extract import get_feature_list,get_feature_map,TraverseDataset2d,TraverseDataset3d,TraverseDataset3d_overlap
from confettii.plot_helper import three_pca_as_rgb_image 
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
class OneDimStatis(widgets.Container):
    def __init__(self, viewer:napari.Viewer, image_layer,featsmap_dict:dict):
        super().__init__()
        self.viewer = viewer
        self.image_layer = image_layer
        self.vsi_feats_layer = self.viewer.add_image(np.zeros(shape=(3,3,3)),name='feats_3pca',rgb=True,opacity=0.6)
        self.feats_map = featsmap_dict 
        self.featsmap_dict = featsmap_dict
        self.rgster_callbacks()
        
    def rgster_callbacks(self):
        # --- Define separate buttons ---
        self.left_length = widgets.Slider(label="left_length",min=24, max = 4000,step=24,value=1200)
        self.right_length = widgets.Slider(label="right_length",min=24, max = 4000,step=24,value=1200)
        self.y = widgets.LineEdit(label = 'y coord', value =994,)
        self.x = widgets.LineEdit(label = 'x coord', value =1088, )
        self.sample_loc = widgets.Container(
            widgets=[self.y, self.x],
            layout='horizontal',
            label='sample_loc'
        )
        self.run_button = widgets.PushButton(
            text = 'similarity_plot'
        )
        self.run_button.clicked.connect(self.on_button_clicked)
        
        
        self.line_type= widgets.ComboBox(
                    choices=['h','v'],
                    label='horizontal or vertical',
                    value = 'h'
                )
        self.feat_type = widgets.ComboBox(
            choices=['gray','mlp','ae','incept'],
            value='mlp',
            label='choose feature'
        )

        self.viewer.mouse_double_click_callbacks.append(self.on_double_click)

        self.extend(
            [
            self.left_length,
            self.right_length,
            self.sample_loc,
            self.line_type,
            self.feat_type,
            self.run_button,
            ]
        )
    def on_double_click(self,viewer,event):

        #get the coordinate of click point--P_i
        mouse_pos=viewer.cursor.position
        P_i=self.image_layer.world_to_data(mouse_pos)
        P_i=np.round(P_i).astype(int)

        #update sample loc
        self.y.value = P_i[0]
        self.x.value = P_i[1]

        self.prepare_for_plot(P_i)
    
    def on_button_clicked(self):
        P_i = np.zeros(shape = 2,dtype=int)
        P_i[0] = int(self.y.value )
        P_i[1] = int(self.x.value)  
        self.prepare_for_plot(P_i)
     
    def prepare_for_plot(self,P_i):

        print(f"data_coords{P_i},{type(P_i[0])= }")
        #update smaple loc

        feats_type = self.feat_type.value
        if feats_type == 'gray':
            feats_map = self.featsmap_dict['gray']
            metric = 'euclidean'
            zoom_factor = 1 
        if feats_type == 'mlp':
            feats_map = self.featsmap_dict['mlp']
            metric = 'cos'
            zoom_factor = 8 
        if feats_type == 'ae':
            feats_map = self.featsmap_dict['ae']
            metric = 'cos'
            zoom_factor = 8 
        if feats_type == 'incept':
            feats_map = self.featsmap_dict['incept']
            metric = 'cos'

        #mapping P_i to P_s and get the correspond sample feature

        H,W = self.image_layer.data.shape
        h_f,w_f,c_f =feats_map.shape
        if feats_type == 'incept':
            zoom_factor = H/h_f
        rgb_feats_map = three_pca_as_rgb_image(feats_map.reshape(-1,c_f),final_image_shape=(h_f,w_f))
        zoomed_rgb_feats_map = zoom(rgb_feats_map,(H/h_f,W/w_f,1),order=1)
        self.vsi_feats_layer.data = zoomed_rgb_feats_map

        P_s = [int(x//zoom_factor) for x in P_i]
        sample_feat = feats_map[P_s[0],P_s[1],:]
        
        #get the deisred llen,rlen value
        llen = int(self.left_length.value)
        rlen = int(self.right_length.value)
        # clip llen and rlen to valid bounds
        llen = min(llen, P_i[1])  # llen should not be more than P_i[1]
        rlen = min(rlen, W - P_i[1] - 1)  # rlen should not cause index to go out of bounds

        #get the row/column ready to be cut into target_feat
        row_feat_i= feats_map[P_s[0],:,:]
        
        scale_factor = W/feats_map.shape[1] #here do not use integer zoom_factor is because some img size are not integer multiples of feature_map 
        row_feat_s = zoom(row_feat_i,zoom=(scale_factor,1),order=1)
        wlb = P_i[1] - llen
        wrb = P_i[1] + rlen + 1
        target_feats_s = row_feat_s[wlb:wrb,:]

        h=240
        ulen = min(h,P_i[0])
        dlen = min (h, H - P_i[0] -1)
        hub = P_i[0] - ulen 
        hdb =  P_i[0] + dlen + 1

        roi = self.image_layer.data[hub:hdb, wlb:wrb] 
        roi_feats = zoomed_rgb_feats_map[hub:hdb, wlb:wrb]

        title = f"feat_{feats_type}_loc{P_i}"

        self.plot_similarity(sample_feat,target_feats_s,llen=llen ,rlen=rlen,roi = roi, roi_feats=roi_feats,metric =metric,ulen=ulen,dlen=dlen,title= title)


    def plot_similarity(self, sample_feat, candidate_feat, llen, rlen, roi,roi_feats, metric, ulen,dlen,title='plot'):
        import matplotlib.pyplot as plt
        import numpy as np

        similarities = []
        for cand in candidate_feat:
            sim = self.compute_similarity(np.array(sample_feat), np.array(cand), metric=metric)
            similarities.append(sim)
        
        if metric == 'euclidean':
            max_dis = max(similarities)
            score = 1 - (similarities/max_dis)
            similarities =  score

        x_positions = np.arange(-llen, rlen + 1)

        fig, axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [0.5, 1]}, sharex=True)

        # Plot similarity curve
        abs_similarities = np.abs(similarities)  # take absolute value
        axs[0].plot(x_positions, abs_similarities, marker='o', markersize=2, label=f'|Similarity| ({metric})')
        axs[0].axvline(x=0, color='r', linestyle='--', label='Sample (x=0)')
        # axs[0].set_ylim([0, 1])  # y-axis from 0 to 1
        axs[0].set_ylabel('Absolute Similarity')
        axs[0].set_title(f'Similarity_Plot_{title}')
        axs[0].legend(loc='upper left')
        axs[0].grid(False)
        axs[0].set_xlim([-llen, rlen + 1])

        # Display ROI image with equal scaling
        extent = [-llen, rlen + 1, -ulen, dlen + 1]
        roi = roi[::-1,:]
        roi_feats = roi_feats[::-1,:]
        axs[1].imshow(roi, cmap='gray', extent=extent)
        axs[1].imshow(roi_feats, alpha=0.39, extent=extent) 

        norm_similarities = np.clip(abs_similarities, 0, 1)

        # Map abs similarities to colors (0 → blue, 1 → red)
        cmap = plt.get_cmap('coolwarm')  # blue to red
        colors = cmap(norm_similarities)  # already 0 to 1 range

        # Overlay thinner color-coded horizontal line (y=0)
        for x, color in zip(x_positions, colors):
            axs[1].plot(x, 0, marker='s', color=color, markersize=1)  # Thinner markers (smaller size)

        # Plot white hollow circle at (0,0)
        axs[1].plot(0, 0, marker='o', markerfacecolor='none', markeredgecolor='white', markersize=3, markeredgewidth=1.5)

        axs[1].set_xlabel('Relative x')
        axs[1].set_ylabel('Relative y')
        # axs[1].set_title('Candidate Overlay with Similarity Line')

        axs[1].set_aspect('equal', adjustable='box')  # Ensure equal scaling
        axs[1].invert_yaxis()

        plt.tight_layout()
        plt.show()

    def compute_similarity(self,vec1, vec2, metric='cos'):
        if metric == 'cos':
            return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0, 0]
        elif metric == 'euclidean':
            e_distances = euclidean_distances(vec1.reshape(1, -1), vec2.reshape(1, -1))[0, 0]  
            return e_distances
        else:
            raise ValueError(f"Unsupported metric: {metric}")


class OneDimStatis_dask_array(widgets.Container):
    def __init__(self, viewer:napari.Viewer, image_layer,vsi_feats_layer,model_dict:dict):
        super().__init__()
        self.viewer = viewer
        self.image_layer = image_layer
        self.vsi_feats_layer = vsi_feats_layer
        self.model_dict = model_dict
        self.rgster_callbacks()
        
    def rgster_callbacks(self):
        # --- Define separate buttons ---
        self.left_length = widgets.Slider(label="left_length",min=24, max = 4000,step=24,value=1200)
        self.right_length = widgets.Slider(label="right_length",min=24, max = 4000,step=24,value=1200)

        self.y = widgets.LineEdit(label = 'y coord', value =0,)
        self.x = widgets.LineEdit(label = 'x coord', value =0, )
        self.sample_loc = widgets.Container(
            widgets=[self.y, self.x],
            layout='horizontal',
            label='sample_loc'
        )
        self.roi_radius = widgets.LineEdit(label='roi_radius', value = 768)

        self.run_button = widgets.PushButton(
            text = 'similarity_plot'
        )
        self.run_button.clicked.connect(self.on_button_clicked)
        
        
        self.line_type= widgets.ComboBox(
                    choices=['h','v'],
                    label='horizontal or vertical',
                    value = 'h'
                )
        self.feat_type = widgets.ComboBox(
            choices=['gray','mlp','inception'],
            value='mlp',
            label='choose feature'
        )
        self.prepare = widgets.PushButton(
            text = 'prepare'
        )
        self.prepare.clicked.connect(self.prepare_feats_map)

        self.viewer.mouse_double_click_callbacks.append(self.on_double_click)

        self.extend(
            [
            self.left_length,
            self.right_length,
            self.sample_loc,
            self.line_type,
            self.feat_type,
            self.run_button,
            self.prepare,
            ]
        )
    def on_double_click(self,viewer,event):

        #get the coordinate of click point--P_i
        mouse_pos=viewer.cursor.position
        P_i=self.image_layer.world_to_data(mouse_pos)
        P_i=np.round(P_i).astype(int)

        #update sample loc
        self.y.value = P_i[1]
        self.x.value = P_i[2]

        self.similarity_plot_pipeline(P_i)
    
    def on_button_clicked(self):
        P_i = np.zeros(shape = 3,dtype=int)
        P_i[1] = int(self.y.value )
        P_i[2] = int(self.x.value)  
        self.similarity_plot_pipeline(P_i)
     
    def prepare_feats_map(self):
        D,H,W = self.image_layer.data.shape
        z = self.viewer.dims.current_step[0]

        print(f"begin prepare for mlp")
        mlp_model = self.model_dict['mlp']
        current = time.time()
        mlp_roi =  self.image_layer.data[z-32:z+32,:,:].compute() 

        roi_save_path = f'/home/confetti/data/t1779/eval_feat_extractor/roi{z-32}_{z+32}.tif'
        tif.imwrite(roi_save_path,mlp_roi)

        print(f"read roi data :{time.time()-current:.3f},roi saved at {roi_save_path}")
        current = time.time()
        # overlap =32
        # dataset = TraverseDataset3d(mlp_roi,overlap=overlap,win_size=(64,1024,1024),verbose=True) 
        # roi_nums = dataset.get_roi_nums()
        # loader = DataLoader(dataset,batch_size=1,shuffle=None,drop_last=False) 
        # #shpae of feats_map should be (h,w,c)
        # mlp_feats_map = get_feature_map('cuda',mlp_model,loader=loader,overlap_i=overlap,roi_nums=roi_nums) 
        stride = 16 
        dataset = TraverseDataset3d(mlp_roi,stride=stride,win_size=(64,64,64),verbose=True) 
        roi_nums = dataset.get_sample_shape()
        loader = DataLoader(dataset,batch_size=512,shuffle=None,drop_last=False) 
        #shpae of feats_map should be (h,w,c)
        mlp_feats_list = get_feature_list('cuda',mlp_model,loader) 
        self.mlp_feats= mlp_feats_list.reshape(roi_nums[1],roi_nums[2],-1)
        mlp_feats_path = '/home/confetti/data/t1779/eval_feat_extractor/mlp_featsmap.pkl'
        with open (mlp_feats_path,'wb') as f:
            pickle.dump(self.mlp_feats,f)
        print(f"extract_feats:{time.time()-current:.3f},{self.mlp_feats.shape= },feats_map saved at{mlp_feats_path}")
        print(f"end prepare for mlp")

        del mlp_roi, dataset, loader, mlp_model
        gc.collect()

        print(f"begin prepare for inception")
        inception_model = self.model_dict['inception']
        roi =  self.image_layer.data[z,:,:].compute() 

        roi_save_path = f'/home/confetti/data/t1779/eval_feat_extractor/roi_z{z}.tif'
        tif.imwrite(roi_save_path,roi)

        inception_extract_layer_name = 'avgpool'
        normalized_img = (((roi -roi.min())/ (roi.max() -roi.min())) * 255).astype(np.uint8)
        rgb_img = np.stack([normalized_img] * 3, axis=-1)
        two_dim_dataset = TraverseDataset2d(rgb_img,stride=16,win_size=128)
        out_shape = two_dim_dataset._get_sample_shape()
        two_dim_loader = DataLoader(two_dim_dataset,batch_size=512,shuffle=None,drop_last=False) 
        current = time.time()
        feats_list = get_feature_list('cuda',inception_model,two_dim_loader,extract_layer_name=inception_extract_layer_name)
        inception_feats_map = feats_list.reshape((*out_shape,-1))
        print(f"end prepare for inception")
        self.inception_feats = inception_feats_map
        incept_feats_path = '/home/confetti/data/t1779/eval_feat_extractor/incept_featsmap.pkl'
        with open (incept_feats_path,'wb') as f:
            pickle.dump(self.inception_feats,f)
        print(f"inception: extract_feats:{time.time()-current:.3f},{self.inception_feats.shape= },feats_map saved at{incept_feats_path}")
        del roi, two_dim_dataset, two_dim_loader,inception_model 
        gc.collect()
        

    def similarity_plot_pipeline(self,P_i):

        print(f"data_coords{P_i},{type(P_i[0])= }")
        #update smaple loc
        D,H,W = self.image_layer.data.shape
        z = self.viewer.dims.current_step[0]

        feats_type = self.feat_type.value
        if feats_type == 'gray':
            zoom_factor = 1
            feats_map = gaussian_filter( self.image_layer.data[z,:,:].compute() ,sigma=12,mode='reflect' )
            feats_map = feats_map[:,:,np.newaxis]
            metric = 'euclidean'
        if feats_type == 'mlp':
            metric = 'cos'
            feats_map = self.mlp_feats

        if feats_type == 'inception':
            metric = 'cos'
            feats_map =self.inception_feats

            print(f"{feats_map.shape= }")

        
        #visulize the feature_map 
        # reshpae to list, pca into 3 component, normlize to 0-1, zoom to img space,set cmap
        h_f, w_f,c_f = feats_map.shape
        current = time.time()
        rgb_feats_map = three_pca_as_rgb_image(feats_map.reshape(-1,c_f),final_image_shape=(h_f,w_f))
        zoomed_rgb_feats_map = zoom(rgb_feats_map,(H/h_f,W/w_f,1),order=1)
        self.vsi_feats_layer.data = zoomed_rgb_feats_map
        print(f"pca: {time.time()-current}")


        #mapping P_i to P_s and get the correspond sample feature
        P_s = [int(x//zoom_factor) for x in P_i]
        sample_feat = feats_map[P_s[1],P_s[2],:]
        
        #get the deisred llen,rlen value
        llen = int(self.left_length.value)
        rlen = int(self.right_length.value)
        # clip llen and rlen to valid bounds
        D,H,W = self.image_layer.data.shape
        llen = min(llen, P_i[2])  # llen should not be more than P_i[1]
        rlen = min(rlen, W - P_i[2] - 1)  # rlen should not cause index to go out of bounds

        #get the row/column ready to be cut into target_feat
        row_feat_i= feats_map[P_s[1],:,:]
        
        scale_factor = W/feats_map.shape[1] #here do not use integer zoom_factor is because some img size are not integer multiples of feature_map 
        row_feat_s = zoom(row_feat_i,zoom=(scale_factor,1),order=1)
        wlb = P_i[2] - llen
        wrb = P_i[2] + rlen + 1
        target_feats_s = row_feat_s[wlb:wrb,:]

        h=240
        ulen = min(h,P_i[1])
        dlen = min (h, H - P_i[1] -1)
        hub = P_i[2] - ulen 
        hdb =  P_i[2] + dlen + 1

        roi = self.image_layer.data[z, hub:hdb, wlb:wrb] 

        title = f"feat_{feats_type}_loc{P_i}"

        self.plot_similarity(sample_feat,target_feats_s,llen=llen ,rlen=rlen,roi = roi, metric =metric,ulen=ulen,dlen=dlen,title= title)

    


class OneDimStatis_roi(widgets.Container):
    def __init__(self, viewer:napari.Viewer, image_layer,model_dict:dict):
        super().__init__()
        self.viewer = viewer
        self.image_layer = image_layer
        self.vsi_feats_layer = self.viewer.add_image(np.zeros(shape=(3,3,3)),name='feats_3pca',rgb=True,opacity=0.6)
        self.model_dict = model_dict
        self.rgster_callbacks()
        
    def rgster_callbacks(self):
        # --- Define separate buttons ---
        self.left_length = widgets.Slider(label="left_length",min=24, max = 4000,step=24,value=1200)
        self.right_length = widgets.Slider(label="right_length",min=24, max = 4000,step=24,value=1200)

        self.y = widgets.LineEdit(label = 'y coord', value =0,)
        self.x = widgets.LineEdit(label = 'x coord', value =0, )
        self.sample_loc = widgets.Container(
            widgets=[self.y, self.x],
            layout='horizontal',
            label='sample_loc'
        )
        self.roi_radius = widgets.LineEdit(label='roi_radius', value = 768)

        self.run_button = widgets.PushButton(
            text = 'similarity_plot'
        )
        self.run_button.clicked.connect(self.on_button_clicked)
        
        
        self.line_type= widgets.ComboBox(
                    choices=['h','v'],
                    label='horizontal or vertical',
                    value = 'h'
                )
        self.feat_type = widgets.ComboBox(
            choices=['gray','mlp','ae','inception'],
            value='mlp',
            label='choose feature'
        )
        self.prepare = widgets.PushButton(
            text = 'prepare and compute feats'
        )
        self.prepare.clicked.connect(self.prepare_feats_map)

        self.viewer.mouse_double_click_callbacks.append(self.on_double_click)

        self.extend(
            [
            self.left_length,
            self.right_length,
            self.sample_loc,
            self.line_type,
            self.feat_type,
            self.prepare,
            self.run_button,
            ]
        )
    def on_double_click(self,viewer,event):

        #get the coordinate of click point--P_i
        mouse_pos=viewer.cursor.position
        P_i=self.image_layer.world_to_data(mouse_pos)
        P_i=np.round(P_i).astype(int)

        #update sample loc
        self.y.value = P_i[1]
        self.x.value = P_i[2]

        self.similarity_plot_pipeline(P_i)
    
    def on_button_clicked(self):
        P_i = np.zeros(shape = 3,dtype=int)
        P_i[1] = int(self.y.value )
        P_i[2] = int(self.x.value)  
        self.similarity_plot_pipeline(P_i)
     
    def prepare_feats_map(self):
        D,H,W = self.image_layer.data.shape
        z = self.viewer.dims.current_step[0]

        print(f"begin prepare for mlp")
        mlp_model = self.model_dict['mlp']
        current = time.time()

        z_dim = D 
        if z_dim >= 64:
            z_center = z_dim // 2
            mlp_roi = self.image_layer.data[z_center - 32 : z_center + 32, :, :]
        else:
            raise ValueError(f"Z-dimension is too small: {z_dim}. Must be at least 64.")
        print(f"read roi data :{time.time()-current:.3f}")

        current = time.time()
        stride = 16 
        dataset = TraverseDataset3d(mlp_roi,stride=16,win_size=(64,64,64),verbose=True) 
        roi_nums = dataset.get_sample_shape()
        loader = DataLoader(dataset,batch_size=512,shuffle=None,drop_last=False) 
        #shpae of feats_map should be (h,w,c)
        mlp_feats_list = get_feature_list('cuda',mlp_model,loader) 
        self.mlp_feats= mlp_feats_list.reshape(roi_nums[1],roi_nums[2],-1)
        print(f"extract_feats:{time.time()-current:.3f}")
        print(f"end prepare for mlp")
         


        print(f"begin prepare for ae")
        ae_model = self.model_dict['ae']
        ae_feats_list = get_feature_list('cuda',ae_model,loader) 
        self.ae_feats = ae_feats_list.reshape(roi_nums[1],roi_nums[2],-1)
        print(f"extract_feats:{time.time()-current:.3f}")
        print(f"end prepare for mlp")

        del mlp_roi, dataset, loader, mlp_feats_list,ae_feats_list,mlp_model, ae_model
        gc.collect()


        

    def similarity_plot_pipeline(self,P_i):

        print(f"data_coords{P_i},{type(P_i[0])= }")
        #update smaple loc
        D,H,W = self.image_layer.data.shape
        z = self.viewer.dims.current_step[0]

        feats_type = self.feat_type.value
        if feats_type == 'gray':
            zoom_factor = 1
            feats_map = gaussian_filter( self.image_layer.data[z,:,:] ,sigma=12,mode='reflect' )
            feats_map = feats_map[:,:,np.newaxis]
            metric = 'euclidean'
        if feats_type == 'mlp':
            metric = 'cos'
            feats_map = self.mlp_feats
            zoom_factor = H/feats_map.shape[0]
        if feats_type == 'ae':
            metric = 'cos'
            feats_map = self.ae_feats
            zoom_factor = H/feats_map.shape[0]
        if feats_type == 'inception':
            metric = 'cos'

            print(f"begin prepare for inception")
            inception_model = self.model_dict['inception']
            roi =  self.image_layer.data[z,:,:]
            inception_extract_layer_name = 'avgpool'
            normalized_img = (((roi -roi.min())/ (roi.max() -roi.min())) * 255).astype(np.uint8)
            rgb_img = np.stack([normalized_img] * 3, axis=-1)
            two_dim_dataset = TraverseDataset2d(rgb_img,stride=8,win_size=128)
            out_shape = two_dim_dataset._get_sample_shape()
            two_dim_loader = DataLoader(two_dim_dataset,batch_size=512,shuffle=None,drop_last=False) 
            current = time.time()
            feats_list = get_feature_list('cuda',inception_model,two_dim_loader,extract_layer_name=inception_extract_layer_name)
            print(f"extracting feats in inception: {time.time()-current:.3f}")
            inception_feats_map = feats_list.reshape((*out_shape,-1))
            print(f"end prepare for inception")
            self.inception_feats = inception_feats_map
            del roi, normalized_img,rgb_img, feats_list,two_dim_loader,inception_model 
            gc.collect()
            feats_map =self.inception_feats
            zoom_factor = feats_map.shape[0]/H

            print(f"{feats_map.shape= }")

        
        #visulize the feature_map 
        # reshpae to list, pca into 3 component, normlize to 0-1, zoom to img space,set cmap
        h_f, w_f,c_f = feats_map.shape
        current = time.time()
        rgb_feats_map = three_pca_as_rgb_image(feats_map.reshape(-1,c_f),final_image_shape=(h_f,w_f))
        zoomed_rgb_feats_map = zoom(rgb_feats_map,(H/h_f,W/w_f,1),order=1)
        self.vsi_feats_layer.data = zoomed_rgb_feats_map
        print(f"pca: {time.time()-current}")


        #mapping P_i to P_s and get the correspond sample feature
        P_s = [int(x/zoom_factor) for x in P_i]
        sample_feat = feats_map[P_s[1],P_s[2],:]
        
        #get the deisred llen,rlen value
        llen = int(self.left_length.value)
        rlen = int(self.right_length.value)
        # clip llen and rlen to valid bounds
        D,H,W = self.image_layer.data.shape
        llen = min(llen, P_i[2])  # llen should not be more than P_i[1]
        rlen = min(rlen, W - P_i[2] - 1)  # rlen should not cause index to go out of bounds

        #get the row/column ready to be cut into target_feat
        row_feat_i= feats_map[P_s[1],:,:]
        
        scale_factor = W/feats_map.shape[1] #here do not use integer zoom_factor is because some img size are not integer multiples of feature_map 
        row_feat_s = zoom(row_feat_i,zoom=(scale_factor,1),order=1)
        wlb = P_i[2] - llen
        wrb = P_i[2] + rlen + 1
        target_feats_s = row_feat_s[wlb:wrb,:]

        h=240
        ulen = min(h,P_i[1])
        dlen = min (h, H - P_i[1] -1)
        hub = P_i[2] - ulen 
        hdb =  P_i[2] + dlen + 1

        roi = self.image_layer.data[z, hub:hdb, wlb:wrb] 

        title = f"feat_{feats_type}_loc{P_i}"

        self.plot_similarity(sample_feat,target_feats_s,llen=llen ,rlen=rlen,roi = roi, metric =metric,ulen=ulen,dlen=dlen,title= title)

    
