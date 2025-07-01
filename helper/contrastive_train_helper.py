import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.manifold import TSNE
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

import random
import pickle
import numpy as np
import tifffile as tif
from helper.image_reader import Ims_Image
from confettii.plot_helper import three_pca_as_rgb_image


class Contrastive_dataset_3d_one_stage(Dataset):
    """
    sample roi in ims file; return positive roi pairs and negative roi pairs for contrastive_learning
    """
    def __init__ (self,ims_path,d_near,num_pairs,n_view = 2,verbose = False):

        self.ims_vol =Ims_Image(ims_path,channel=0) 
        level = 0
        D,H,W= self.ims_vol.rois[level][3:]
        d_near = int(d_near) 

        margin = 10 
        # Generate random (x, y, z) locations within the given range
        lz, hz = d_near + margin +int(D//4),  int(D*3/4) - d_near - margin
        ly, hy = d_near + margin +int(H//4),  int(H*3/4) - d_near - margin
        lx, hx = d_near + margin +int(W//4),  int(W//2) - d_near - margin

        self.loc_lst = np.stack([
            np.random.randint(lz, hz, size=num_pairs),
            np.random.randint(ly, hy, size=num_pairs),
            np.random.randint(lx, hx, size=num_pairs)
        ], axis=1)

        self.sample_num = num_pairs
        self.all_near_shifts = generate_sphereshell__shifts(R= d_near,r= 24 ,dims=3)
        self.n_view =n_view


    def __len__(self):
        return self.sample_num
    
    def __getitem__(self,idx):

        z, y, x = self.loc_lst[idx].T    # Unpack coordinates
        roi = self.ims_vol.from_roi(coords=(z,y,x,64,64,64))
        roi = roi.astype(np.float32)
        roi=torch.from_numpy(roi)
        roi=torch.unsqueeze(roi,0)
        # for each call, the positive pair is resampled within the near_shifts range, 
        # maybe fix the positive pair will be better for stable training
        pair_locs = [self.positve_pair_loc_generate([z,y,x]) for _ in range(self.n_view -1)]
        pair_roi = [self.get_roi_given_loc(pair_loc) for pair_loc in pair_locs]
        res = [roi]
        for pair in pair_roi:
            res.append(pair)

        #res: x1,neigb1(x1),neigb2(x1),..,neigbN(x1)
        return res

    def get_roi_given_loc(self,loc):
        z, y, x = loc   # Unpack coordinates
        roi = self.ims_vol.from_roi(coords=(z,y,x,64,64,64))
        roi = roi.astype(np.float32)
        roi=torch.from_numpy(roi)
        roi=torch.unsqueeze(roi,0)
        return roi 

    def positve_pair_loc_generate(self,loc):
        shift = random.choice(self.all_near_shifts)
        return loc+shift


class Contrastive_dataset_3d_fix_paris(Dataset):
    """
    fix the positive pair at the init

    """
    def __init__(self, feats_map, d_near, num_pairs, n_view=2, verbose=False):
        D, H, W, C = feats_map.shape
        self.feats_map = feats_map
        d_near = int(d_near)
        margin = 10

        # Generate random (z, y, x) locations within the safe volume
        lz, hz = d_near + margin, D - d_near - margin
        ly, hy = d_near + margin, H - d_near - margin
        lx, hx = d_near + margin, W - d_near - margin

        self.loc_lst = np.stack([
            np.random.randint(lz, hz, size=num_pairs),
            np.random.randint(ly, hy, size=num_pairs),
            np.random.randint(lx, hx, size=num_pairs)
        ], axis=1)

        self.sample_num = num_pairs
        self.n_view = n_view
        self.all_near_shifts = generate_sphereshell__shifts(R=d_near, r=0, dims=3)

        # Fix positive pairs at initialization
        self.fixed_positive_pairs = [
            [self.positve_pair_loc_generate(self.loc_lst[i]) for _ in range(n_view - 1)]
            for i in range(self.sample_num)
        ]

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        z, y, x = self.loc_lst[idx]
        feat = self.feats_map[z, y, x, :]  # Shape: (C)
        feat = torch.from_numpy(feat)

        # Use fixed positive pairs
        pair_locs = self.fixed_positive_pairs[idx]
        pair_feats = [torch.from_numpy(self.get_feats_given_loc(loc, self.feats_map)) for loc in pair_locs]

        # res: x1, neigb1(x1), neigb2(x1), ..., neigbN(x1)
        return [feat] + pair_feats

    def get_feats_given_loc(self, loc, feat_map):
        z, y, x = loc
        return feat_map[z, y, x, :]

    def positve_pair_loc_generate(self, loc):
        shift = random.choice(self.all_near_shifts)
        loc = np.array(loc)
        new_loc = loc + shift
        return np.clip(new_loc, [0, 0, 0], np.array(self.feats_map.shape[:3]) - 1)  # Ensure valid bounds

class Contrastive_dataset_3d(Dataset):
    """
    Supports both 4D (D, H, W, C) and 3D (H, W, C) feature maps.
    """
    def __init__(self, feats_map, d_near, num_pairs, n_view=2, verbose=False,margin=10):
        self.feats_map = feats_map
        self.dims = feats_map.ndim - 1  # 3D (volumetric) or 2D (single slice)
        self.verbose = verbose
        self.n_view = n_view
        d_near = int(d_near)
        

        if self.dims == 3:
            D, H, W, C = feats_map.shape
            lx, hx = d_near + margin, D - d_near - margin
            ly, hy = d_near + margin, H - d_near - margin
            lz, hz = d_near + margin, W - d_near - margin
            self.loc_lst = np.stack([
                np.random.randint(lx, hx, size=num_pairs),
                np.random.randint(ly, hy, size=num_pairs),
                np.random.randint(lz, hz, size=num_pairs)
            ], axis=1)
        elif self.dims == 2:
            H, W, C = feats_map.shape
            ly, hy = d_near + margin, H - d_near - margin
            lz, hz = d_near + margin, W - d_near - margin
            self.loc_lst = np.stack([
                np.random.randint(ly, hy, size=num_pairs),
                np.random.randint(lz, hz, size=num_pairs)
            ], axis=1)
        else:
            raise ValueError("Feature map must be either 4D (D, H, W, C) or 3D (H, W, C).")

        self.sample_num = num_pairs
        self.all_near_shifts = generate_sphereshell__shifts(R=d_near, r=0, dims=self.dims)

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        loc = self.loc_lst[idx]
        feat = torch.from_numpy(self.get_feats_given_loc(loc)).float()
        pair_locs = [self.positive_pair_loc_generate(loc) for _ in range(self.n_view - 1)]
        pair_feats = [torch.from_numpy(self.get_feats_given_loc(pl)).float() for pl in pair_locs]
        return [feat] + pair_feats

    def get_feats_given_loc(self, loc):
        if self.dims == 3:
            z, y, x = loc
            return self.feats_map[z, y, x, :]
        elif self.dims == 2:
            y, x = loc
            return self.feats_map[y, x, :]

    def positive_pair_loc_generate(self, loc):
        shift = random.choice(self.all_near_shifts)
        return loc + shift

class Contrastive_dataset_multiple_2d(Dataset):
    def __init__ (self,feats_map,d_near,n_view = 2,verbose = False):

        N,C,H,W = feats_map.shape
        self.feats_maps = feats_map
        self.feats_map_shape = (C,H,W)
        self.data_length = N
        self.margin = 10

        d_near = int(d_near//(2**3*0.5))

        self.all_near_shifts = generate_sphereshell__shifts(R= d_near,r= 0, dims=2)
        self.n_view =n_view


    def __len__(self):
        return self.data_length
    
    def __getitem__(self,idx):

        C,H,W = self.feats_map_shape
        margin = self.margin
        ith_feat_map = self.feats_maps[idx]  # Shape: (C)
        y = random.randint(margin, H - margin - 1)
        x = random.randint(margin, W - margin - 1)
        sampled_feat = ith_feat_map[:,y,x]

        sampled_feat = torch.from_numpy(sampled_feat)
        # for each call, the positive pair is resampled within the near_shifts range, 
        # maybe fix the positive pair will be better for stable training
        pair_locs = [self.positve_pair_loc_generate([y,x]) for _ in range(self.n_view -1)]
        pair_feats = [ith_feat_map[:,pair_loc[0],pair_loc[1]] for pair_loc in pair_locs]
        res = [sampled_feat]
        for pair in pair_feats:
            res.append(torch.from_numpy(pair).float())

        #res: x1,neigb1(x1),neigb2(x1),..,neigbN(x1)
        return res

    def positve_pair_loc_generate(self,loc):
        shift = random.choice(self.all_near_shifts)
        return loc+shift



# You also need to modify your generate_sphereshell__shifts function to accept a `dims` argument.
def generate_sphereshell__shifts(R, r=0, dims=3):
    """Generate integer shifts within a sphere shell (radius R, inner radius r)."""
    shifts = []
    ranges = [range(-R, R+1)] * dims
    for shift in np.array(np.meshgrid(*ranges)).T.reshape(-1, dims):
        norm = np.linalg.norm(shift)
        if r < norm <= R:
            shifts.append(shift)
    return np.array(shifts)


import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, filters=[24, 18, 12, 8]):
        super(MLP, self).__init__()
        
        layers = []
        for in_features, out_features in zip(filters[:-1], filters[1:]):
            layers.append(nn.Linear(in_features, out_features))
        
        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        x = self.layers[-1](x)  # Last layer, no activation
        return x / x.norm(p=2, dim=-1, keepdim=True)

def cos_loss(features,n_views,pos_weight_ratio=5,enhanced =False):

    #labels for positive pairs
    N = features.shape[0]
    labels = torch.cat([torch.arange(int(N//n_views)) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    
    cos_similarity_matrix = torch.matmul(features, features.T)
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool)
    labels = labels[~mask].view(labels.shape[0], -1)
    # print(f"{labels.shape=}")
    # print(f"{features.shape=}")
    # print(f"{mask.shape=}")
    cos_similarity_matrix = cos_similarity_matrix[~mask].view(cos_similarity_matrix.shape[0], -1)

    # select and combine multiple positives and the negatives
    pos_coses = cos_similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    neg_coses = cos_similarity_matrix[~labels.bool()].view(cos_similarity_matrix.shape[0], -1)

    if enhanced:
        mean_pos_cose = pos_coses.abs().mean()
        mean_neg_cose = neg_coses.abs().mean()

        # Filter pos_coses: keep only those with abs > mean_pos_cose
        filtered_pos = pos_coses[pos_coses.abs() > mean_pos_cose]
        if filtered_pos.numel() > 0:
            pos_loss = ((filtered_pos - 1) ** 2).mean()
        else:
            pos_loss = torch.tensor(0.0, device=pos_coses.device)

        # Filter neg_coses: keep only those with abs < mean_neg_cose
        filtered_neg = neg_coses[neg_coses.abs() < mean_neg_cose]
        if filtered_neg.numel() > 0:
            neg_loss = (filtered_neg ** 2).mean()
        else:
            neg_loss = torch.tensor(0.0, device=neg_coses.device)
    
    else:
        pos_loss = ((pos_coses -1 )**2).mean()
        neg_loss = (neg_coses**2).mean()

    pos_weight = (pos_weight_ratio)/(pos_weight_ratio+1)
    neg_weight = (1)/(pos_weight_ratio+1)

    
    # pos_loss =(torch.exp( torch.abs((pos_coses-1)/T) ) -1 ).mean()
    # neg_loss = (torch.exp( torch.abs((neg_coses)/T) ) -1 ).mean() 
    return pos_loss*pos_weight +neg_loss*neg_weight, pos_coses.mean(),neg_coses.mean()


def cos_loss_topk(features,n_views,pos_weight_ratio=5,only_pos=False):
    #labels for positive pairs
    N = features.shape[0]
    labels = torch.cat([torch.arange(int(N//n_views)) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    
    cos_similarity_matrix = torch.matmul(features, features.T)
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool)
    labels = labels[~mask].view(labels.shape[0], -1)

    cos_similarity_matrix = cos_similarity_matrix[~mask].view(cos_similarity_matrix.shape[0], -1)

    # select and combine multiple positives and the negatives
    pos_coses = cos_similarity_matrix[labels.bool()].view(labels.shape[0], -1) #shape (N,n_view-1)
    neg_coses = cos_similarity_matrix[~labels.bool()].view(cos_similarity_matrix.shape[0], -1) # shape (N, N - n_view)

    k_pos = min(int(n_views/4),1)
    k_neg = int(N/10)
    filtered_pos, topk_pos_indices = torch.topk(pos_coses.abs(),k_pos,dim=-1) #shape :N,k_pos
    if only_pos:
        filtered_neg = neg_coses
    else:
        filtered_neg, topk_neg_indices = torch.topk(neg_coses.abs(),k_neg,dim=-1,largest=False) #shape: N,k_neg
    filtered_pos_loss = ((filtered_pos - 1) ** 2).mean()
    filtered_neg_loss = (filtered_neg ** 2).mean()
 
    pos_weight = (pos_weight_ratio)/(pos_weight_ratio+1)
    neg_weight = (1)/(pos_weight_ratio+1)
    
    return pos_weight*filtered_pos_loss + neg_weight+filtered_neg_loss, pos_coses.mean(),neg_coses.mean()





def compute_ncc_map(loc_idx, encoded, shape):
    ncc_list =[]
    for  idx in loc_idx :
        att = encoded @ encoded[idx]
        ncc_list.append(att.reshape(shape))
    return ncc_list





def plot_ncc_maps(img_lst,loc_idx_lst,locations,writer, tag="ncc_plot", step=0,ncols=4,fig_size=4):
    num_plots = len(img_lst)
    ncols = ncols
    nrows = int(np.ceil(num_plots/ ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_size*ncols,  fig_size*nrows))

    for i, ax in enumerate(axes.flat):
        if i < num_plots:
            ax.imshow(img_lst[i], cmap='viridis')
            ax.plot(locations[loc_idx_lst[i], 1], locations[loc_idx_lst[i], 0], '*r')
        ax.axis('off')
    plt.tight_layout()
    # Save plot to TensorBoard

    writer.add_figure(tag, fig, global_step=step)
    plt.close(fig)

def plot_pca_maps(img_lst,writer, tag="pca_plot", step=0,ncols=4,fig_size=4):
    num_plots = len(img_lst)
    ncols = ncols
    nrows = int(np.ceil(num_plots/ ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_size*ncols,  fig_size*nrows))

    for i, ax in enumerate(axes.flat):
        if i < num_plots:
            ax.imshow(img_lst[i])
        ax.axis('off')
    plt.tight_layout()
    # Save plot to TensorBoard

    writer.add_figure(tag, fig, global_step=step)
    plt.close(fig)




def tsne_plot(encoded, labels, ax, title='tsne'):
    # Filter out label 0
    labels = np.array(labels)
    mask = labels != 0
    filtered_encoded = encoded[mask]
    filtered_labels = labels[mask]

    tsne_model = TSNE(n_components=2, perplexity=20, random_state=42)
    reduced_data = tsne_model.fit_transform(filtered_encoded)
    
    scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], s=1.2, c=filtered_labels, cmap='tab10')
    ax.legend(*scatter.legend_elements(), title="Digits")
    ax.set_title(title)

def tsne_grid_plot(encoded_list, labels_list, writer, tag='tsne_grid', step=0):
    num_plots = len(encoded_list)
    fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 6))
    
    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable for a single plot
    
    for i, (encoded, labels) in enumerate(zip(encoded_list, labels_list)):
        tsne_plot(encoded, labels, axes[i], title=f'{tag}_{i}')
    
    writer.add_figure(tag, fig, global_step=step)
    plt.close(fig)

def get_vsi_eval_data(img_no_list =[1,3,4,5]):
    eval_feats_path = '/home/confetti/data/wide_filed/test_hp_raw_feats.pkl'
    eval_label_path = '/home/confetti/data/wide_filed/test_hp_label.tif'
    eval_img = tif.imread("/home/confetti/data/wide_filed/test_hp_img.tif")


    with open(eval_feats_path,'rb') as f:
        eval_feats = pickle.load(f)
    
    eval_labels = tif.imread(eval_label_path)
    eval_labels = zoom(eval_labels,(1/16),order=0)
    eval_labels = eval_labels.ravel()
    print(f"eval_labels:{eval_labels.shape}")

    H,W = [int(x//16)for x in eval_img.shape]
    coords = [[i, j] for i in range(H) for j in range(W)]
    coords = np.array(coords)
    eval_datas ={}
    eval_datas['img']=eval_img
    eval_datas['label']=eval_labels
    eval_datas['feats']=eval_feats
    eval_datas['locs']=coords
    # sample point idx for ncc plot
    eval_datas['loc_idx']=[940,2162,4895,8768,10020]
    eval_datas = [eval_datas]

    return eval_datas 

def get_t11_eval_data(E5,img_no_list =[1,3,4,5],ncc_seed_point = True):
    eval_datas = []
    # === eval feats and ncc_points and labels ===#
    # Set the path prefix based on E5 flag
    if E5:
        prefix = "/share/home/shiqiz/data"
    else:
        prefix = "/home/confetti/data"

    # === eval feats and ncc_points and labels ===#
    for img_no in img_no_list:
        eval_label_path = f"{prefix}/t1779/test_data_part_brain/human_mask_{img_no:04d}.tif"
        vol = tif.imread(f"{prefix}/t1779/test_data_part_brain/{img_no:04d}.tif")
        z_slice = vol[int(vol.shape[0]//2),:,:]

        eval_dic = {}
        eval_dic['img'] = vol
        eval_dic['label'] = tif.imread(eval_label_path)
        eval_dic['z_slice'] = z_slice
        eval_datas.append(eval_dic)

    # sample point idx for ncc plot
    if ncc_seed_point:
        eval_datas[0]['loc_idx']=[940,2162,978,4024,3607]
        eval_datas[1]['loc_idx']=[1015,1883,968,3459,4152]
        # eval_datas[2]['loc_idx']=[6907, 1982, 2367, 2193,2296]
        eval_datas[2]['loc_idx']=[6907, 1982, 376, 4692, 2459]
        eval_datas[3]['loc_idx']=[1779, 3453, 3653, 3223,978]
    return eval_datas


def get_rm009_eval_data(E5,):
    eval_datas = []
    # === eval feats and ncc_points and labels ===#
    # Set the path prefix based on E5 flag
    if E5:
        prefix = "/share/home/shiqiz/data"
    else:
        prefix = "/home/confetti/data"

    # === eval feats and ncc_points and labels ===#
    for img_no in [1,2]:
        vol = tif.imread(f"{prefix}/rm009/seg_valid/{img_no:04d}.tif")
        mask = tif.imread(f"{prefix}/rm009/seg_valid/{img_no:04d}_human_mask.tif")
        
        z_slice = vol[int(vol.shape[0]//2),:,:]

        eval_dic = {}
        eval_dic['img'] = vol
        eval_dic['label'] = mask 
        eval_dic['z_slice'] = z_slice
        eval_datas.append(eval_dic)


    return eval_datas


def valid_from_roi(model,it,eval_data,writer):
    "eval from roi"
    model.eval()
    ncc_valid_imgs=[]
    pca_img_lst=[]
    ncc_seedpoints_idx_lsts =[]
    tsne_encoded_feats_lst4tsne=[]
    tsne_label_lst4tsne=[]
    for idx,data_dic in enumerate(eval_data):
        roi = data_dic['img']
        label= data_dic['label']
        idxes = data_dic.get('loc_idx', [])

        input = torch.from_numpy(roi).unsqueeze(0).unsqueeze(0).float().to('cuda')
        outs = model(input).cpu().detach().squeeze().numpy() #C*H*W    D will equals to 1 and is ignored
        feats_map = np.moveaxis(outs,0,-1) #H*W*C
        H,W,C = feats_map.shape
        feats_list = feats_map.reshape(-1, C)

        #rgb_plot
        rgb_img = three_pca_as_rgb_image(feats_list,(H,W))
        pca_img_lst.append(rgb_img)

        #tsne_plot
        tsne_encoded_feats_lst4tsne.append(feats_list)

        #ncc_plot
        if len(idxes) > 0:
            ncc_lst = compute_ncc_map(idxes,feats_list, shape = (H,W))
            ncc_valid_imgs.extend(ncc_lst)
            ncc_seedpoints_idx_lsts.extend(idxes)
            rows, cols = np.indices((H, W))
            locations = np.stack([rows.ravel(), cols.ravel()], axis=1)

        zoom_factor = [ y/x for y,x in zip([H,W],label.shape)]
        label = zoom(label,zoom_factor,order=0)
        tsne_label_lst4tsne.append(label.ravel())


    tsne_grid_plot(tsne_encoded_feats_lst4tsne,tsne_label_lst4tsne,writer,tag=f'tsne',step =it)
    plot_pca_maps(pca_img_lst, writer=writer, tag=f"pca",step=it,ncols=idx+1)

    if len(idxes) > 0: 
        plot_ncc_maps(ncc_valid_imgs, ncc_seedpoints_idx_lsts,locations,writer=writer, tag=f"ncc",step=it,ncols=len(idxes))




def valid_from_feats(model,it,eval_data,writer):
    "eval from ae_feats, older version"
    model.eval()

    ncc_valid_imgs=[]
    ncc_seedpoints_idx_lsts =[]
    tsne_encoded_feats_lst4tsne=[]
    tsne_label_lst4tsne=[]
    for idx,data_dic in enumerate(eval_data):
        feats=data_dic['feats']
        locations =data_dic['locs'] 
        labels= data_dic['label']
        idxes = data_dic['loc_idx']

        feats = torch.from_numpy(feats).float().to('cuda')
        encoded = model(feats)
        encoded = encoded.detach().cpu().numpy()
        n = int(np.sqrt(len(encoded.shape[0])))
        ncc_lst = compute_ncc_map(idxes,encoded,(n,n))
        ncc_valid_imgs.extend(ncc_lst)
        ncc_seedpoints_idx_lsts.extend(idxes)
        tsne_encoded_feats_lst4tsne.append(encoded)
        tsne_label_lst4tsne.append(labels)

    tsne_grid_plot(tsne_encoded_feats_lst4tsne,tsne_label_lst4tsne,writer,tag=f'tsne',step =it)
    plot_ncc_maps(ncc_valid_imgs, ncc_seedpoints_idx_lsts,locations,writer=writer, tag=f"ncc",step=it,ncols=len(idxes))




