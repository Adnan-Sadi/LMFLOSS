# -*- coding: utf-8 -*-
"""
@author: Adnan-Sadi
"""
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights
from torchvision.models import ResNet50_Weights
from torchvision.models import DenseNet121_Weights

"""
Function for getting the pre-trained EfficientNetV2 model

Args:
    out_ftrs: number of output features of the FC layer
    freeze_wts: boolean value for freezing the weights of the Conv layers,
                defaults to False
   
Returns:
    model_ef: EfficientNetV2 model pretrained on IMAGENET1K_V1 weights
"""
def get_efficientnetv2(out_ftrs, freeze_wts=False):
    
    model_ef = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    
    if(freeze_wts == True):
        for param in model_ef.parameters():
            param.requires_grad = False
    
    num_ftrs = model_ef.classifier[1].in_features
    model_ef.classifier[1] = nn.Linear(num_ftrs, out_ftrs, bias = True)
    
    return model_ef

"""
Function for getting the pre-trained ResNet50 model

Args:
    out_ftrs: number of output features of the FC layer
    freeze_wts: boolean value for freezing the weights of the Conv layers,
                defaults to False
   
Returns:
    model_rs: ResNet50 model pretrained on IMAGENET1K_V2 weights
"""
def get_resnet50(out_ftrs, freeze_wts=False):
    model_rs = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    if(freeze_wts == True):
        for param in model_rs.parameters():
            param.requires_grad = False
            
    num_ftrs = model_rs.fc.in_features
    model_rs.fc = nn.Linear(num_ftrs, out_ftrs, bias = True)
    
    return model_rs

"""
Function for getting the pre-trained DenseNet-121 model

Args:
    out_ftrs: number of output features of the FC layer
    freeze_wts: boolean value for freezing the weights of the Conv layers,
                defaults to False
   
Returns:
    model_dns: DenseNet-121 model pretrained on IMAGENET1K_V1 weights
"""
def get_densenet121(out_ftrs, freeze_wts=False):
    model_dns = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    
    if(freeze_wts == True):
        for param in model_dns.parameters():
            param.requires_grad = False
            
    num_ftrs = model_dns.classifier.in_features
    model_dns.classifier = nn.Linear(num_ftrs, out_ftrs, bias = True)
    
    return model_dns