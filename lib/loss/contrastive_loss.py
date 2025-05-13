import torch
import torch.nn as nn
from ._info_nec_loss import _info_nce_loss

class ContrastiveLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.contrastive_loss=_info_nce_loss
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, preds, labels):
         
        logits, labels =  self.contrastive_loss(self.cfg,preds,labels)
        loss = self.loss_fn(logits, labels)

        return loss


def get_loss(cfg):
	
    return ContrastiveLoss(cfg)
