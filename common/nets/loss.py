import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from config import cfg

class distance_loss(nn.Module):
    def __init__(self):
        super(distance_loss, self).__init__()
        self.loss =nn.L1Loss()
        self.softmax = nn.Softmax(dim=0)
    def forward(self, gt_loss_term, pred_loss_term, center_obj_bbox,index_selection):
        batch_size,num_j,xy=gt_loss_term.shape
        distance_gt=torch.abs(gt_loss_term-center_obj_bbox)/2.
        distance_pred=torch.abs(pred_loss_term-center_obj_bbox)/2.
        loss_term=torch.zeros((batch_size,1))
        for num, (gt,pred,valid) in enumerate(zip(distance_gt,distance_pred,index_selection)):
            gt_val_xy,pred_val_xy=gt[valid],pred[valid]
            gt_val_total=gt_val_xy[:,0]+gt_val_xy[:,1]
            pred_val_total=pred_val_xy[:,0]+pred_val_xy[:,1]
            loss_term[num]=self.loss(self.softmax(-gt_val_total),self.softmax(-pred_val_total))

        return loss_term