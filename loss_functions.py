# -*- coding: utf-8 -*-
"""
@author: Adnan-Sadi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#BWCCE Loss
def bwcce_loss(img_per_cls, device):
    K = len(img_per_cls)-1
    print()
    print(img_per_cls)
    
    beta_values = [(1 - (x / sum(img_per_cls))) / K for x in img_per_cls]
    print(beta_values)
    print(sum(beta_values))
    
    bwcce = nn.CrossEntropyLoss(weight=torch.Tensor(beta_values).to(device))
    return bwcce
    
# LDAM Loss
class LDAMLoss(nn.Module):
    """
    Constructor

    Args:
        class_num_list: class dependent margin values (calculated using the get_mlist() funcion in utils.py)
        max_m: maximum margin value, defaults to 0.5
        s: scaling factor, defaults to 30
    """
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = cls_num_list
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)
    

# Focal Loss
"""
Function for getting Focal loss

Args:
    device: the device the loss function is moved to
    alpha: assigned weights for each class, defaults to None
    gamma: constant, defaults to 1.5
    
Returns:
    focal: the focal loss function
"""
def focal_loss(device, alpha=None, gamma = 1.5):
    focal = torch.hub.load(
    'adeelh/pytorch-multi-class-focal-loss',
    model='focal_loss', 
    alpha= alpha,
    gamma=gamma,
    reduction='mean',
    device=device,
    dtype=torch.float32,
    force_reload=False,
    verbose=False
    )
    
    return focal


# LMF Loss
class LMFLoss(nn.Module):
    
    """
    Constructor

    Args:
        ldam: LDAM loss function
        focal: Focal loss function
        alpha: Weight factor of LDAM loss, defaults to 1
        beta: Weight factor of focal loss, defaults to 1
        
    """
    def __init__(self, ldam, focal, alpha=1, beta=1):
        super(LMFLoss, self).__init__()
        self.ldam = ldam
        self.focal = focal
        self.alpha = alpha
        self.beta = beta

    
    """
    forward function

    Args:
        outputs: outputs from model
        labels: corresponding labels
        
    Returns:
        lmf: the lmf loss value
    """
    def forward(self, outputs, labels):
        loss_ldam = self.ldam(outputs, labels)
        loss_focal = self.focal(outputs, labels)
        lmf = (self.alpha * loss_ldam) + (self.beta * loss_focal)
        
        return lmf
    