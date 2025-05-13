from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pickle

import random
import numpy as np
import tifffile as tif


class Contrastive_dataset(Dataset):
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

def cos_loss(features,batch_size,n_views,pos_weight_ratio=5,eps=1e-9,T=0.1):

    #labels for positive pairs
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    
    features = F.normalize(features +eps, dim=1)

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
    
    pos_loss = ((pos_coses -1 )**2).mean()
    neg_loss = (neg_coses**2).mean()
    pos_weight = (pos_weight_ratio)/(pos_weight_ratio+1)
    neg_weight = (1)/(pos_weight_ratio+1)

    
    # pos_loss =(torch.exp( torch.abs((pos_coses-1)/T) ) -1 ).mean()
    # neg_loss = (torch.exp( torch.abs((neg_coses)/T) ) -1 ).mean() 
    return pos_loss*pos_weight +neg_loss*neg_weight, pos_coses.mean(),neg_coses.mean()







def compute_ncc_map(loc_idx, encoded, locations):
    img_length = int(np.sqrt(encoded.shape[0]))
    ncc_list =[]
    for  idx in loc_idx :
        att = encoded @ encoded[idx]
        img = np.zeros((img_length, img_length))
        
        for i in range(len(att)):
            img[locations[i][0], locations[i][1]] = att[i]
        ncc_list.append(img)
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

def get_eval_data(img_no_list =[1,3,4,5]):
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

def valid(model,it,eval_data,writer):
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
        ncc_lst = compute_ncc_map(idxes,encoded,locations)
        ncc_valid_imgs.extend(ncc_lst)
        ncc_seedpoints_idx_lsts.extend(idxes)
        tsne_encoded_feats_lst4tsne.append(encoded)
        tsne_label_lst4tsne.append(labels)


    tsne_grid_plot(tsne_encoded_feats_lst4tsne,tsne_label_lst4tsne,writer,tag=f'tsne',step =it)
    plot_ncc_maps(ncc_valid_imgs, ncc_seedpoints_idx_lsts,locations,writer=writer, tag=f"ncc",step=it,ncols=len(idxes))


