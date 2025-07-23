
import numpy as np
from sklearn.neighbors import BallTree
import torch
from confettii.plot_helper import three_pca_as_rgb_image

import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import time
import matplotlib.pyplot as plt
import os

from pathlib import Path

import os
import matplotlib.pyplot as plt
from pathlib import Path

class HTMLFigureLogger:
    def __init__(self, log_dir='html_log', html_name='index.html', comment="logged_figures"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        html_base      = Path(html_name).stem
        self.images_dir = self.log_dir / f"{html_base}_images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.html_path = self.log_dir / html_name
        self.comment   = comment
        self._init_html()
        self.entries = []

    def _init_html(self):
        with open(self.html_path, "w") as f:
            f.write("<html><head><title>Figure Log</title></head><body>\n")
            f.write(f"<h1>{self.comment}</h1>\n")
            f.write("<style>img {width: 200px; margin: 5px;} "
                    ".row {display: flex; flex-wrap: wrap;}</style>\n")

    def add_figure(self, tag, fig, global_step):
        img_filename = f"{tag}_{global_step}.png".replace("/", "_")
        img_path     = self.images_dir / img_filename
        fig.savefig(img_path, dpi=300,bbox_inches='tight')
        plt.close(fig)
        self.entries.append((img_path, img_filename, tag, global_step))

    def finalize(self):
        with open(self.html_path, 'a') as f:
            for i, (img_path, fname, tag, step) in enumerate(self.entries):
                if i % 6 == 0:
                    if i > 0:    # close previous row
                        f.write("</div>\n")
                    f.write("<div class='row'>\n")

                # ---- key line: make src *relative* to html file ----
                rel_src = os.path.relpath(img_path, start=self.html_path.parent)
                f.write(
                    f"<div><img src='{rel_src}' alt='{tag} step {step}'><br>"
                    f"{tag} (Step {step})</div>\n"
                )

            f.write("</div>\n</body></html>\n")


def simple_eval(out,epoch,img_logger,writer,img_shape,tag='pca'):
    eval_outs = out.detach().cpu().squeeze().numpy() #N*C
    rgb_img = three_pca_as_rgb_image(eval_outs,img_shape)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.axis('off')
    ax.imshow(rgb_img) 
    img_logger.add_figure(tag, fig, global_step=epoch)
    writer.add_figure(tag, fig, global_step=epoch)

def train_demo(agg_feats,locations,d_near,d_far,mlp_model,
               num_pairs,writer,img_logger,img_shape,num_epochs,valid_very_epoch,shuffle_pairs_epoch_num,percentile,model_save_dir):
    # build near_pairs_all via ball_tree consume ~2s when N is 5*10^4
    current = time.time()
    near_pairs_all_balltree = generate_all_near_pairs_balltree(locations, d_near)
    print(f"total_near_pairs_num:{len(near_pairs_all_balltree)}")
    print(f"generate_near using time {time.time()-current}")
    current = time.time()
    device = 'cuda'
    optimizer = optim.Adam(mlp_model.parameters(), lr=0.0005) 
    scheduler = ExponentialLR(optimizer, gamma=1)

    #prepare positive and negtive pairs
    local_pairs, far_pairs = sample_pairs_balltree(locations, near_pairs_all_balltree, d_far, num_pairs)

    #agg_feats shape: (roi_num,N,64)
    for epoch in range(num_epochs): 
        # one roi one batch, each roi is of same shape
        for i, batch in enumerate(agg_feats):
            batch = torch.from_numpy(batch).float().to(device)
            optimizer.zero_grad() 
            out = mlp_model(batch) 
            # loss,pos_cos,neg_cos= loss_fn(out, local_pairs, far_pairs,enhanced=enhanced)
            loss,pos_cos,neg_cos= loss_fn_percentile(out, local_pairs, far_pairs,percentile=percentile)
            loss.backward() 
            optimizer.step() 
            #whole dataset as a batch
            # scheduler.step()
        #end of one epoch
        print(f"epoch:  {epoch}, loss: {loss}")

        writer.add_scalar('Loss/train', loss.item(), epoch)
        writer.add_scalar('pos_cos',pos_cos.item(), epoch)
        writer.add_scalar('neg_cos',neg_cos.item(), epoch)

        if (epoch) % valid_very_epoch ==0: 
            simple_eval(out,epoch+1,img_logger,writer,img_shape=img_shape,tag=f'pca')

        if epoch % shuffle_pairs_epoch_num==0: 
            # update local and far paris
            local_pairs, far_pairs = sample_pairs_balltree(locations, near_pairs_all_balltree, d_far, num_pairs)
        
        # if (epoch+1) % 100 == 0:
        #     model_path = os.path.join(model_save_dir, f'model_epoch_{epoch+1}.pth')
        #     torch.save(mlp_model.state_dict(), model_path)
        #     print(f"Model saved at epoch {epoch +1} to {model_path}")
    
    img_logger.finalize()
    del optimizer,mlp_model,agg_feats,locations




def loss_fn(out, local_pairs, far_pairs,pos_weight_ratio=2,enhanced =False):
    pos_coses = (out[local_pairs[:, 0]] * out[local_pairs[:, 1]]).sum(axis=-1) 
    neg_coses = (out[far_pairs[:, 0]] * out[far_pairs[:, 1]]).sum(axis=-1)

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

    return pos_loss*pos_weight + neg_loss*neg_weight, pos_coses.mean(),neg_coses.mean()


def loss_fn_percentile(out, local_pairs, far_pairs,pos_weight_ratio=2,percentile =0.2):
    pos_coses = (out[local_pairs[:, 0]] * out[local_pairs[:, 1]]).sum(axis=-1) 
    neg_coses = (out[far_pairs[:, 0]] * out[far_pairs[:, 1]]).sum(axis=-1)

    if percentile:

        pos_abs = pos_coses.abs()
        neg_abs = neg_coses.abs()

        if percentile ==0.5:
            pos_thresh = torch.mean(pos_abs)
            neg_thresh = torch.mean(neg_abs)
        else:
            pos_thresh = torch.quantile(pos_abs, percentile )
            neg_thresh = torch.quantile(neg_abs, percentile )

        # Filter pos_coses: keep only those with abs > mean_pos_cose
        filtered_pos = pos_coses[pos_coses.abs() > pos_thresh]
        if filtered_pos.numel() > 0:
            pos_loss = ((filtered_pos - 1) ** 2).mean()
        else:
            pos_loss = torch.tensor(0.0, device=pos_coses.device)

        # Filter neg_coses: keep only those with abs < mean_neg_cose
        filtered_neg = neg_coses[neg_coses.abs() < neg_thresh]
        if filtered_neg.numel() > 0:
            neg_loss = (filtered_neg ** 2).mean()
        else:
            neg_loss = torch.tensor(0.0, device=neg_coses.device)
    
    else:
        pos_loss = ((pos_coses -1 )**2).mean()
        neg_loss = (neg_coses**2).mean() 

    pos_weight = (pos_weight_ratio)/(pos_weight_ratio+1)
    neg_weight = (1)/(pos_weight_ratio+1)

    return pos_loss*pos_weight + neg_loss*neg_weight, pos_coses.mean(),neg_coses.mean()

def loss_fn_soft_near(out, local_pairs, far_pairs):
    gamma =0.09
    cosine_near= (out[local_pairs[:, 0]] * out[local_pairs[:, 1]]).sum(axis=-1)  
    loss1 = torch.exp(- (torch.abs(cosine_near-0.5))**2 / gamma)
    loss2 = ((out[far_pairs[:, 0]] * out[far_pairs[:, 1]]).sum(axis=-1))**2 
    return (loss1.mean() + loss2.mean())/2.0


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