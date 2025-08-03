#%%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_skeletonize(x, thresh_width=5):
    is_3d = (x.dim() == 5)
    for _ in range(thresh_width):
        if is_3d:
            p1 = F.max_pool3d(x * -1, (3, 1, 1), stride=1, padding=(1, 0, 0)) * -1
            p2 = F.max_pool3d(x * -1, (1, 3, 1), stride=1, padding=(0, 1, 0)) * -1
            p3 = F.max_pool3d(x * -1, (1, 1, 3), stride=1, padding=(0, 0, 1)) * -1
            min_pool = torch.min(torch.min(p1, p2), p3)
            contour = F.relu(F.max_pool3d(min_pool, 3, 1, 1) - min_pool)
        else:
            p1 = F.max_pool2d(x * -1, (3, 1), stride=1, padding=(1, 0)) * -1
            p2 = F.max_pool2d(x * -1, (1, 3), stride=1, padding=(0, 1)) * -1
            min_pool = torch.min(p1, p2)
            contour = F.relu(F.max_pool2d(min_pool, 3, 1, 1) - min_pool)
        x = F.relu(x - contour)
    return x

def positive_intersection(center_line, vessel):
    clf = center_line.view(center_line.size(0), center_line.size(1), -1)
    vf = vessel.view(vessel.size(0), vessel.size(1), -1)
    intersection = (clf * vf).sum(-1)
    return (intersection.sum(0) + 1e-12) / (clf.sum(-1).sum(0) + 1e-12)

def soft_cldice(pred, target):
    target_skeleton = soft_skeletonize(target)
    cl_pred = soft_skeletonize(pred)

    clrecall = positive_intersection(target_skeleton, pred)  # ClRecall
    clacc = positive_intersection(cl_pred, target)           # ClPrecision

    recall = positive_intersection(target, pred)
    acc = positive_intersection(pred, target)

    show = False 
    if show:
        np_cl_pred = np.squeeze(cl_pred.cpu().detach().numpy())[0]
        np_cl_targe = np.squeeze(target_skeleton.cpu().detach().numpy())[0]
        dpi =100
        fig, axes = plt.subplots(2, 2 ,figsize=( 8*np_cl_pred.shape[1] / dpi, 8*np_cl_pred.shape[0] / dpi), dpi=dpi)

        axes[0,0].imshow(np_cl_pred, cmap='tab20',interpolation='nearest')
        axes[0,0].set_title('cl_pred')
        axes[0,0].axis('off')

        axes[0,1].imshow(np_cl_targe, cmap='tab20',interpolation='nearest')
        axes[0,1].set_title('cl_target')
        axes[0,1].axis('off')

        axes[1,0].imshow(np.squeeze(pred.cpu().detach().numpy())[0], cmap='tab20',interpolation='nearest')
        axes[1,0].set_title('pred')
        axes[1,0].axis('off')

        axes[1,1].imshow(np.squeeze(target.cpu().detach().numpy())[0], cmap='tab20',interpolation='nearest')
        axes[1,1].set_title('target')
        axes[1,1].axis('off')

        plt.tight_layout()
        plt.show() 


    return clrecall[0], clacc[0], recall[0], acc[0]

class SoftclDiceLoss(nn.Module):
    def __init__(self, thresh_width=5):
        super().__init__()
        self.thresh_width = thresh_width

    def forward(self, logits, target, mask=None):
        """
        logits:   (B, 1, H, W) or (B, 1, D, H, W)
        target: (B, 1, H, W) or (B, 1, D, H, W)
        mask:   (B, 1, H, W) or (B, 1, D, H, W), optional, values >= 0 are included
        """
        if logits.shape[2] ==1:
            logits = logits.squeeze(2)
            target = target.squeeze(2)
            mask   =  mask.squeeze(2)

        pred = torch.sigmoid(logits)

        if mask is not None:
            valid = mask.float()
            pred = pred * valid
            target = target * valid

        clrecall, clacc, recall, acc = soft_cldice(pred, target)
        cldice = (2. * clrecall * clacc) / (clrecall + clacc + 1e-12)
        return 1 - cldice

def get_loss(args=None, model=None):
    return SoftclDiceLoss()

#%%
# import tifffile as tif
# import numpy as np
# from scipy.ndimage import zoom
# label = tif.imread("/home/confetti/data/rm009/boundary_seg/valid_bnd_masks/0002.tif")
# roi = tif.imread("/home/confetti/data/rm009/boundary_seg/valid_rois/0002.tiff")
# mask = tif.imread("/home/confetti/data/rm009/boundary_seg/valid_masks/0002.tiff")
# label = np.squeeze(label) 
# mask = np.squeeze(mask)
# print(f"{label.shape=}, {roi.shape= }, {mask.shape= }")
# label = zoom(label,0.25,order=0)
# dpi =100
# plt.figure(figsize=( 4*label.shape[1] / dpi, 4*label.shape[0] / dpi), dpi=dpi)
# plt.imshow(label,cmap='tab10',interpolation='nearest')


# # %%
# input_label = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float() #B*C*H*W
# cl_label = soft_skeletonize(input_label).cpu().detach().squeeze().numpy()
# print(cl_label.shape)
# plt.figure(figsize=( 4*label.shape[1] / dpi, 4*label.shape[0] / dpi), dpi=dpi)
# plt.imshow(cl_label,cmap='tab10',interpolation='nearest')

#%%