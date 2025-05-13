from magicgui import magicgui,widgets
import napari
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class OneDimStatis(widgets.Container):
    def __init__(self, viewer:napari.Viewer, image_layer,featsmap_dict:dict):
        super().__init__()
        self.viewer = viewer
        self.image_layer = image_layer
        self.feats_map = featsmap_dict 
        self.featsmap_dict = featsmap_dict
        self.rgster_callbacks()
        
    def rgster_callbacks(self):
        # --- Define separate buttons ---
        self.left_length = widgets.Slider(label="left_length",min=24, max = 4000,step=24,value=1200)
        self.right_length = widgets.Slider(label="right_length",min=24, max = 4000,step=24,value=1200)
        self.line_type= widgets.ComboBox(
                    choices=['h','v'],
                    label='horizontal or vertical',
                    value = 'h'
                )
        self.feat_type = widgets.ComboBox(
            choices=['gray','mlp'],
            value='gray',
        )

        self.viewer.mouse_double_click_callbacks.append(self.on_double_click)
        self.extend(
            [
            self.left_length,
            self.right_length,
            self.line_type,
            self.feat_type,
            ]
        )
    def on_double_click(self,viewer,event):

        #adjust parameters based on feat_type
        feats_type = self.feat_type.value
        if feats_type == 'gray':
            zoom_factor = 1
            feats_map = self.featsmap_dict['gray']
            metric = 'euclidean'
        if feats_type == 'mlp':
            zoom_factor = 8
            feats_map = self.featsmap_dict['mlp']
            metric = 'cos'

        #get the coordinate of click point--P_i
        mouse_pos=viewer.cursor.position
        P_i=self.image_layer.world_to_data(mouse_pos)
        P_i=np.round(P_i).astype(int)
        print(f"data_coords{P_i}")

        #mapping P_i to P_s and get the correspond sample feature
        P_s = [int(x//zoom_factor) for x in P_i]
        sample_feat = feats_map[P_s[0],P_s[1],:]
        
        #get the deisred llen,rlen value
        llen = int(self.left_length.value)
        rlen = int(self.right_length.value)
        # clip llen and rlen to valid bounds
        H,W = self.image_layer.data.shape
        llen = min(llen, P_i[1])  # llen should not be more than P_i[1]
        rlen = min(rlen, W - P_i[1] - 1)  # rlen should not cause index to go out of bounds

        #get the row/column ready to be cut into target_feat
        row_feat_i= feats_map[P_s[0],:,:]
        row_feat_s = zoom(row_feat_i,zoom=(zoom_factor,1),order=1)
        wlb = P_i[1] - llen
        wrb = P_i[1] + rlen + 1
        target_feats_s = row_feat_s[wlb:wrb,:]

        h=240
        ulen = min(h,P_i[0])
        dlen = min (h, H - P_i[0] -1)
        hub = P_i[0] - ulen 
        hdb =  P_i[0] + dlen + 1

        roi = self.image_layer.data[hub:hdb, wlb:wrb] 

        self.plot_similarity(sample_feat,target_feats_s,llen=llen ,rlen=rlen,roi = roi, metric =metric,ulen=ulen,dlen=dlen)


    def compute_similarity(self,vec1, vec2, metric='cos'):
        if metric == 'cos':
            return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0, 0]
        elif metric == 'euclidean':
            e_distances = euclidean_distances(vec1.reshape(1, -1), vec2.reshape(1, -1))[0, 0]  
            return e_distances
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    def plot_similarity(self, sample_feat, candidate_feat, llen, rlen, roi, metric, ulen,dlen):
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
        axs[0].set_ylim([0, 1])  # y-axis from 0 to 1
        axs[0].set_ylabel('Absolute Similarity')
        axs[0].set_title('Similarity Plot')
        axs[0].legend()
        axs[0].grid(True)
        axs[0].set_xlim([-llen, rlen + 1])

        # Display ROI image with equal scaling
        extent = [-llen, rlen + 1, -ulen, dlen + 1]
        axs[1].imshow(roi, cmap='gray', extent=extent)

        norm_similarities = np.clip(abs_similarities, 0, 1)

        # Map abs similarities to colors (0 → blue, 1 → red)
        cmap = plt.get_cmap('coolwarm')  # blue to red
        colors = cmap(norm_similarities)  # already 0 to 1 range

        # Overlay thinner color-coded horizontal line (y=0)
        for x, color in zip(x_positions, colors):
            axs[1].plot(x, 0, marker='s', color=color, markersize=1)  # Thinner markers (smaller size)

        # Plot white hollow circle at (0,0)
        axs[1].plot(0, 0, marker='o', markerfacecolor='none', markeredgecolor='white', markersize=3, markeredgewidth=1.5)

        axs[1].set_xlabel('Relative Location')
        axs[1].set_ylabel('Height')
        axs[1].set_title('Candidate Overlay with Similarity Line')

        axs[1].set_aspect('equal', adjustable='box')  # Ensure equal scaling
        axs[1].invert_yaxis()

        plt.tight_layout()
        plt.show()
