import torch
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


def cos_loss_topk(features,n_views,pos_weight_ratio=5,topk=False,only_pos=False):
    """
    update: 2025/08/12
    three mode: norml, topk, topk_only_pos
    the returned pos and neg loss are the filtered one, the previous are whole original loss
    """
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

    if topk:
        k_pos = min(int(n_views/4),1)
        k_neg = int(N/10)

        filtered_pos, topk_pos_indices = torch.topk(pos_coses.abs(),k_pos,dim=-1) #shape :N,k_pos
        if only_pos:
            filtered_neg = neg_coses
        else:
            filtered_neg, topk_neg_indices = torch.topk(neg_coses.abs(),k_neg,dim=-1,largest=False) #shape: N,k_neg
        pos_loss = ((filtered_pos - 1) ** 2).mean()
        neg_loss = (filtered_neg ** 2).mean()
    else:
        pos_loss = ((pos_coses- 1) ** 2).mean()
        neg_loss = (neg_coses** 2).mean()

 
    pos_weight = (pos_weight_ratio)/(pos_weight_ratio+1)
    neg_weight = (1)/(pos_weight_ratio+1)
    
    return pos_weight*pos_loss + neg_weight+neg_loss, pos_coses.mean(),neg_coses.mean()


