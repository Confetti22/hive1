import torch
import torch.nn.functional as F

def compute_feature_list(ori_features:torch.Tensor, psd_label:torch.Tensor , rare_class_ratio_thres=0.06):
    """
    directly compute the average of features for each class, do not split the features in a class into two groups`:w
    pros: matrix computation, efficient, 
        if the input data is augmentated to form two inputs pair i and j, 
        then can use this funciton to compute contrastive loss
    parameters:
    features  : B*C*Z*Y*X
    psd_label : B*1*Z*Y*X
    

    discard the class that with ratio lower than rare_class_thres
    rare_class_ratio
    map the psd_label from 0 to N-1, generate a lable_lst, the index is new class label
    for each class in class label
    """
    #reshape features into(B*Z*Y*X)*C, for each pixel, normolize the feature vector
    #reshape   labels into (B*Z*Y*X)
    ori_features=ori_features.permute(1,0,2,3,4)
    ori_features=ori_features.reshape(-1,ori_features.shape[0])
    feature_map_flat = F.normalize(ori_features, dim=1) 

    psd_label_flat=psd_label.view(-1)

    # Compute class proportions
    total_pixels = psd_label_flat.numel()
    unique_classes, counts = torch.unique(psd_label_flat, return_counts=True)
    proportions = counts.float() / total_pixels
    valid_classes = unique_classes[proportions >= rare_class_ratio_thres]

    # Create a one-hot encoding of the mask for valid classes
    valid_mask = (psd_label_flat.unsqueeze(0) == valid_classes.unsqueeze(1)).float()  # Shape: (num_valid_classes, H*W)

    # Multiply the one-hot mask with the feature map and sum
    valid_pixel_sums = valid_mask @ feature_map_flat  # Shape: (num_valid_classes, C)

    # Count the number of pixels per valid class
    valid_pixel_counts = valid_mask.sum(dim=1)  # Shape: (num_valid_classes,)

    # Compute the average feature vectors
    avg_features = valid_pixel_sums / valid_pixel_counts.unsqueeze(1)  # Shape: (num_valid_classes, C)

    # Return results as a dictionary
    return avg_features 


def compute_feature_list_split2(cfg,ori_features:torch.Tensor, psd_label:torch.Tensor ):
    """
    randomly sample half of the features and average to be f_i, the rest to be g_i

    discard the class that with ratio lower than rare_class_ratio_thres
    rare_class_ratio_thres should be compute dynamiclly,
    min_pixel_num = min_lenght(um) / voxel_size(um)
    min_voxel_num = min_pixel_num**3
    rare_class_ratio_thres = min_voxel_num / totoal_voxel_count

    map the psd_label from 0 to N-1, generate a lable_lst, the index is new class label
    for each class in class label

    inputs:
    features  : B*C*Z*Y*X
    psd_label : B*1*Z*Y*X
    
    outputs:
    a feature_list:Tensor of shape 2N*C (N is the number of valid classes)
    """
    #reshape features into(B*Z*Y*X)*C, for each pixel, normolize the feature vector
    #reshape   labels into (B*Z*Y*X)
    B,C,Z,Y,X=ori_features.shape
    ori_features=ori_features.permute(1,0,2,3,4)
    ori_features=ori_features.reshape(-1,ori_features.shape[0])
    feature_map_flat = F.normalize(ori_features, dim=1) 

    psd_label_flat=psd_label.view(-1)


    
    # Compute class proportions
    total_pixels = psd_label_flat.numel()
    unique_classes, counts = torch.unique(psd_label_flat, return_counts=True)
    #convert to Long to support the following indexing operation: unique_class[proportions >= thres]
    unique_classes=unique_classes.to(cfg.SYSTEM.DEVICE).long() 
    counts.to(cfg.SYSTEM.DEVICE)
    
    proportions = counts.float() / total_pixels
    
    #compute rare_class_ratio_thres
    min_voxel_num = (cfg.DATASET.min_valid_texture_length / cfg.DATASET.voxel_size)**3
    rare_class_ratio_thres = min_voxel_num / total_pixels


    thres=torch.tensor(rare_class_ratio_thres,device=cfg.SYSTEM.DEVICE, dtype=torch.float32)


    mask = proportions >= thres  # This should produce a boolean tensor
    # print("unique_classes dtype, device:", unique_classes.dtype, unique_classes.device)
    # print("mask dtype, device:", mask.dtype, mask.device)

    valid_classes = unique_classes[mask]


    feature_list1 = torch.zeros(size=(len(unique_classes),C)).to(cfg.SYSTEM.DEVICE)
    feature_list2 = torch.zeros(size=(len(unique_classes),C)).to(cfg.SYSTEM.DEVICE)

    for idx,cls in enumerate(valid_classes):
        # Get indices of pixels belonging to the current class
        cls_indices = torch.nonzero(psd_label_flat == cls, as_tuple=False).squeeze(1)

        # Ensure there are enough samples
        num_samples = cls_indices.numel()

        # Randomly sample 50% of the indices for f_i
        num_half = num_samples // 2
        perm1 = torch.randperm(num_samples)
        sample1_indices = cls_indices[perm1[:num_half]]

        # f_i as the average of sampled features
        f_i = feature_map_flat[sample1_indices].mean(dim=0)

        # g_i as the average of remaining features (the rest of the class)
        remaining_indices = cls_indices[perm1[num_half:]]
        g_i = feature_map_flat[remaining_indices].mean(dim=0)

        # Add the results to the feature list
        feature_list1[idx]=f_i
        feature_list2[idx]=g_i
      # Stack the results into a single tensor of shape (2N, C)
    return torch.cat((feature_list1,feature_list2), dim=0)



def _info_nce_loss( cfg,ori_features:torch.Tensor, psd_label:torch.Tensor ):
    """
    parameters:
    features  : B*C*Z*Y*X
    psd_label : B*1*Z*Y*X

    first getnerate features list based on psd_label
    then compute similarity matrix
    then compute logits
    """
    features=compute_feature_list_split2(cfg,ori_features,psd_label)
    N=int(features.shape[0]//2)
 
    similarity_matrix = torch.matmul(features, features.T)

    labels = torch.cat([torch.arange(N) for i in range(cfg.LOSS.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(cfg.SYSTEM.DEVICE)


    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(cfg.SYSTEM.DEVICE)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(cfg.SYSTEM.DEVICE)

    logits = logits / cfg.LOSS.temperature
    logits.requires_grad_()
    return logits, labels
