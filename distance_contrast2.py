#%%
import sys
import os
# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)
#%%
from lib.arch.ae import build_final_model,load_compose_encoder_dict,build_encoder_model,load_encoder2encoder
from config.load_config import load_cfg
from torchsummary import summary
from confettii.plot_helper import grid_plot_list_imgs
import time
from torchsummary import summary
from confettii.feat_extract import get_feature_list, get_feature_map
device ='cuda'
print(f'{os.getcwd()}=')
args = load_cfg('config/rm009.yaml')

avg_pool = None 
args.avg_pool_size = [avg_pool]*3
args.avg_pool_padding =  False

# the old result on rm009 did not load the correct encoder feats
# cmpsd_model = build_final_model(args)
# cmpsd_model.eval().to(device)
# cnn_ckpt_pth = '/home/confetti/e5_workspace/hive/rm009_ae_out/weights/test_rm009/Epoch_1451.pth'
cnn_ckpt_pth = '/home/confetti/data/weights/rm009_3d_ae_best.pth'
# mlp_ckpt_pth ='/home/confetti/e5_workspace/hive/contrastive_run_rm009/batch4096_nview2_pos_weight_2_mlp[96, 48, 24, 12]_d_near1/model_epoch_2049.pth'
# load_compose_encoder_dict(cmpsd_model,cnn_ckpt_pth,mlp_ckpt_pth,dims=args.dims)

encoder_model = build_encoder_model(args,dims=3) 
encoder_model.eval().to(device)
load_encoder2encoder(encoder_model,cnn_ckpt_pth)
summary(encoder_model,(1,*args.input_size))
print(encoder_model)


#%%
import torch 
import numpy as np 
import matplotlib.pyplot as plt 
import pickle
from scipy.spatial.distance import pdist, squareform
import time
import tifffile as tif

vol = tif.imread('/home/confetti/data/rm009/seg_valid/_0001.tif')
mask = tif.imread('/home/confetti/data/rm009/seg_valid/_0001_human_mask_3d.tif')
print(f"{vol.shape= },{mask.shape= }")

# vol = pad_to_multiple_of_unit(vol,unit=zoom_factor) 
print(f"after padding: {vol.shape= }")

#crop the cortex roi 
# vol = vol[:,432:1100,206:1329]
# mask = mask[:,432:1100,206:1329]
# mask_slice = mask[32]

#%% get different_resolution feature_map
import torch.nn as nn
activation = {}
def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

hook1 = encoder_model.conv_in.register_forward_hook(getActivation('layer1'))
hook2 = encoder_model.down_layers[0].register_forward_hook(getActivation('layer2'))
inputs = torch.from_numpy(vol).float().unsqueeze(0).to(device)
outs = np.squeeze(encoder_model(inputs).cpu().detach().numpy())
l1_feats_map =  np.squeeze(activation['layer1'].cpu().detach().numpy())
l2_feats_map =  np.squeeze(activation['layer2'].cpu().detach().numpy())
print(f"{l1_feats_map.shape= },{l2_feats_map.shape= },{outs.shape= }")

hook1.remove()
hook2.remove()

for k in (2,4,8):
pool = nn.AvgPool3d(kernel_size=[k]*3, stride=1,padding=pad)
#%%


if len(features.shape)==2:
    features=features[np.newaxis,:]
agg_feats = features

print(f"feats.shape :{agg_feats.shape}")
print(f"locations.shape :{locations.shape}")




d_near= 3  # 6*8 = 48 um
d_far= 32  # 64*8 = 512 um
num_pairs=100000
shuffle_pairs_epoch_num = 100
num_epochs = 30000


#%% finding neighbor points via thresholding distrance matrix
# works fine when N <= 10^4, did not work when N=10^5
import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
def compute_pair_indices_by_dmatrix(locations, d_near=4, d_far=32):
    """Compute all local and far pairs based on given distance thresholds."""
    start_time = time.time()
    dist = pdist(locations, metric='euclidean')
    dist_matrix = squareform(dist)
    print(f"Distance matrix computation time: {time.time() - start_time:.4f} seconds")
    
    start_time = time.time()
    local_pairs_all = np.array(np.where(dist_matrix <= d_near)).T
    far_pairs_all = np.array(np.where(dist_matrix > d_far)).T
    print(f"Finding pairs computation time: {time.time() - start_time:.4f} seconds")
    
    return local_pairs_all, far_pairs_all

def sample_pairs_dmatrix(local_pairs_all, far_pairs_all, num_pairs=100000):
    """Sample num_pairs of local and far pairs randomly."""
    idx_local = np.random.randint(0, len(local_pairs_all), num_pairs)
    idx_far = np.random.randint(0, len(far_pairs_all), num_pairs)
    
    local_pairs = local_pairs_all[idx_local, :]
    far_pairs = far_pairs_all[idx_far, :]
    
    return local_pairs, far_pairs


# current=time.time()
# local_pairs_all_dmatrix, far_pairs_all_dmatrix = compute_pair_indices_by_dmatrix(locations,d_near=d_near,d_far=d_far)
# print(f"create distance matrix consume time {time.time()-current}")
# local_pairs, far_pairs = sample_pairs_dmatrix(local_pairs_all_dmatrix, far_pairs_all_dmatrix, num_pairs=num_pairs)

#%%
import numpy as np
from sklearn.neighbors import BallTree

def generate_all_near_pairs_balltree(locations, d_near):
    """
    Generate all local pairs within a distance threshold using BallTree.

    Parameters:
        locations (np.ndarray): Shape (N, 2), array of 2D coordinates.
        d_near (float): Distance threshold for local pairs.

    Returns:
        near_pairs (np.ndarray): Shape (M, 2), all local pairs found.
    """
    tree = BallTree(locations, metric='euclidean')
    near_pairs_list = []
    
    for i, neighbors in enumerate(tree.query_radius(locations, r=d_near, return_distance=False)):
        for j in neighbors:
            if i < j:  # Avoid duplicate pairs
                near_pairs_list.append((i, j))
    
    return np.array(near_pairs_list)

def sample_pairs_balltree(locations, near_pairs, d_far, num_pairs):
    """
    Sample `num_pairs` local and far pairs from precomputed near pairs.

    Parameters:
        locations (np.ndarray): Shape (N, 2), array of 2D coordinates.
        near_pairs (np.ndarray): Precomputed near pairs.
        d_far (float): Distance threshold for far pairs.
        num_pairs (int): Number of pairs to sample for each category.

    Returns:
        local_pairs (np.ndarray): Shape (num_pairs, 2), sampled local pairs.
        far_pairs (np.ndarray): Shape (num_pairs, 2), sampled far pairs.
    """
    N = locations.shape[0]
    
    # Sample local pairs
    if len(near_pairs) > num_pairs:
        local_pairs = near_pairs[np.random.choice(len(near_pairs), num_pairs, replace=False)]
    else:
        local_pairs = near_pairs
    
    # Randomly sample far pairs
    far_pairs_list = []
    while len(far_pairs_list) < num_pairs:
        idx1 = np.random.randint(0, N, num_pairs * 2)  # Oversample to increase valid samples
        idx2 = np.random.randint(0, N, num_pairs * 2)
        valid_mask = np.linalg.norm(locations[idx1] - locations[idx2], axis=1) > d_far
        sampled = list(zip(idx1[valid_mask], idx2[valid_mask]))
        far_pairs_list.extend(sampled)
    
    far_pairs = np.array(far_pairs_list[:num_pairs])  # Truncate to desired size
    
    return local_pairs, far_pairs

# build near_pairs_all via ball_tree consume ~2s when N is 5*10^4
current = time.time()
near_pairs_all_balltree = generate_all_near_pairs_balltree(locations, d_near)
print(f"total_near_pairs_num:{len(near_pairs_all_balltree)}")
print(f"generate_near using time {time.time()-current}")
current = time.time()
# d_matrix is a symmetric matrix, near_pairs_all_balltree does not contain duplicated ones and piars composed by itself
#  subtract diagonal elemets and all the near_pairs numbeer, the rest is far_pairs numbers
# print(f"total_far_pairs_num:{ (N**2 -N -2*len(near_pairs_all_balltree))/2}")

local_pairs, far_pairs = sample_pairs_balltree(locations, near_pairs_all_balltree, d_far, num_pairs)
print(f"sampling using time {time.time()-current}")
current = time.time()


# In[37]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(96, 48)  # Input layer to first hidden layer
        self.fc2 = nn.Linear(48, 24)  # First hidden layer to second hidden layer
        self.fc3 = nn.Linear(24, 12)  # Second hidden layer to third hidden layer
        self.fc4 = nn.Linear(12, 12)  # Second hidden layer to third hidden layer
        self.relu = nn.ReLU()  # Activation function

    def forward(self, x):
        # Forward pass through the network
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # No activation on the output layer
        return x / x.norm(p=2, dim=-1, keepdim=True)

def loss_fn(out, local_pairs, far_pairs):
    loss1 = ((out[local_pairs[:, 0]] * out[local_pairs[:, 1]]).sum(axis=-1) - 1)**2 
    loss2 = ((out[far_pairs[:, 0]] * out[far_pairs[:, 1]]).sum(axis=-1))**2 
    return (loss1.mean() + loss2.mean())/2.0

def loss_fn_soft_near(out, local_pairs, far_pairs):
    gamma =0.09
    cosine_near= (out[local_pairs[:, 0]] * out[local_pairs[:, 1]]).sum(axis=-1)  
    loss1 = torch.exp(- (torch.abs(cosine_near-0.5))**2 / gamma)
    loss2 = ((out[far_pairs[:, 0]] * out[far_pairs[:, 1]]).sum(axis=-1))**2 
    return (loss1.mean() + loss2.mean())/2.0


# In[38]:

device = 'cuda'
model = MLP().to(device)

x = torch.from_numpy(features).float().to(device)
out = model(x)
print(x.shape)
print(out.shape)

optimizer = optim.Adam(model.parameters(), lr=0.001) 
scheduler = ExponentialLR(optimizer, gamma=1)

# In[64]:



# local_pairs, far_pairs = sample_pairs_dmatrix(local_pairs_all_dmatrix, far_pairs_all_dmatrix, num_pairs=num_pairs)
local_pairs, far_pairs = sample_pairs_balltree(locations, near_pairs_all_balltree, d_far, num_pairs)


#agg_feats shape: (roi_num,N,64)
for epoch in range(num_epochs): 
    # one roi one batch, each roi is of same shape
    for i, batch in enumerate(agg_feats):
        batch = torch.from_numpy(batch).float().to(device)

        optimizer.zero_grad() 
        out = model(batch) 
        loss = loss_fn(out, local_pairs, far_pairs)
        loss.backward() 
        optimizer.step() 
        #whole dataset as a batch
        # scheduler.step()
    #end of one epoch

    if epoch % shuffle_pairs_epoch_num==0: 
        print(f"epoch:  {epoch}, loss: {loss}")
        # update local and far paris
        local_pairs, far_pairs = sample_pairs_balltree(locations, near_pairs_all_balltree, d_far, num_pairs)
        # local_pairs, far_pairs = sample_pairs_dmatrix(local_pairs_all_dmatrix, far_pairs_all_dmatrix, num_pairs=num_pairs)



#%%
train_encoded = out.detach().cpu().numpy()
#%%

img_no = 3
eval_feats_path = f"/home/confetti/data/t1779/test_data_part_brain/{img_no:04d}_feats.pkl"
eval_loactions_path =f"/home/confetti/data/t1779/test_data_part_brain/{img_no:04d}_indexes.pkl"
eval_label_path =f"/home/confetti/data/t1779/test_data_part_brain/{img_no:04d}_labels.pkl"

with open(eval_feats_path,'rb') as f:
    eval_feats = pickle.load(f)
with open(eval_loactions_path,'rb') as f:
    eval_locations= pickle.load(f)
with open(eval_label_path,'rb') as f:
    eval_labels= pickle.load(f)

eval_feats_tensor = torch.from_numpy(eval_feats).float().to(device)
model.eval()
eval_encoded = model(eval_feats_tensor)
eval_encoded = eval_encoded.detach().cpu().numpy()
#%%
def cos_theta_plot(idx,encoded,locations):
    # att = encoded @ encoded[idx].unsqueeze(1)
    att = encoded @ encoded[idx]
    img_length = int(np.sqrt(encoded.shape[0]))
    img = np.zeros((img_length, img_length))

    for i in range(len(att)): 
        img[locations[i, 0], locations[i, 1]] = att[i]

    plt.figure(figsize=(8,8))
    plt.plot(locations[idx, 1], locations[idx, 0], '*r')
    plt.imshow(img)
    plt.colorbar()

def norm_data(data):
    mean_data=np.mean(data)
    std_data=np.std(data, ddof=1)
    #return (data-mean_data)/(std_data*np.sqrt(data.size-1))
    return (data-mean_data)/(std_data)

def ncc(data0, data1):
    return (1.0/(data0.size-1)) * np.sum(norm_data(data0)*norm_data(data1))

def ncc_plot(idx,data_lst,locations):
    # att = encoded @ encoded[idx].unsqueeze(1)
    template = data_lst[idx]
    ncc_score = np.zeros(shape=(len(data_lst)))
    for i, sample in enumerate(data_lst):
        ncc_score[i] = ncc(sample,template)
    image_size = int(np.sqrt(data_lst.shape[0]))
    ncc_score = ncc_score.reshape((image_size,image_size))
    plt.figure(figsize=(8,8))
    plt.plot(locations[idx, 1], locations[idx, 0], '*r')
    plt.imshow(ncc_score)
    plt.colorbar()

#%%
eval3_loc_idx =[1015,1883,968,3459,4152]
eval1_loc_idx =[940,2162,978,4024,3607]
print(f"in train_dataset")
for idx in eval3_loc_idx:
    cos_theta_plot(idx,train_encoded,locations=locations)
#%%
eval3_loc_idx =[1015,1883,968,3459,4152]
eval1_loc_idx =[940,2162,978,4024,3607]
print(f"in eval_dataset")
for idx in eval3_loc_idx:
    cos_theta_plot(idx,eval_encoded,locations=eval_locations)
#%%
for idx in eval3_loc_idx:
    ncc_plot(idx,eval_encoded,locations=eval_locations)

#eval3_loc_computed from the following
# loc1 = 10*93 +85
# loc2 = 20*93 +23
# loc3 = 10*93 +38
# loc4 = 37*93 +18
# loc5 = 44*93 +60




# %%

#%%





def plot_2d_scatter_point(x,y,label,title='plot'):
# Plot the 2D projection
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y,s=1.2,c=label,cmap='tab10')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(*scatter.legend_elements(), title="Digits")
    plt.title(title)
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train_encoded[:, 0], train_encoded[:, 1], train_encoded[:, 2], c=labels,cmap='tab10',marker='.')  # 'c' is color, 'marker' is the shape of points

# Set labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Show the plot
plt.show()


# In[76]:


import numpy as np
import matplotlib.pyplot as plt
import umap

# Example data: replace this with your actual N x 12 array
data = train_encoded.numpy() # 100 samples with 12 features each

# Perform UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)  
reduced_data = umap_model.fit_transform(data)

plot_2d_scatter_point(reduced_data[:,0],reduced_data[:,1],label=labels,title='after mlp visualize by umap')



# In[ ]:




