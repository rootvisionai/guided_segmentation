# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 12:46:34 2022

@author: tekin.evrim.ozmermer
"""

import torch


def l2_norm(x):
    input_size = x.size()
    buffer = torch.pow(x, 2)

    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)

    _output = torch.div(x, norm.unsqueeze(1))

    output = _output.view(input_size)

    return output

def reshape_feat_map(fm):
    fm = fm.view(fm.shape[0], fm.shape[1], fm.shape[2] * fm.shape[3]) # (1,64,N) -> (1,N,64)
    fm = fm.permute(0, 2, 1)
    return fm

def norm_feature_map_size(fm, target_size = (1024, 1024)):
    return torch.nn.functional.interpolate(
        fm,
        size=target_size,
        mode='nearest'
    )

def norm_mask_size(mask, target_size = (56, 56)):
    return torch.nn.functional.interpolate(
        mask, 
        size=target_size,
        mode='nearest'
    )

def convert_3d_2d(x):
    return x.reshape(x.shape[0] * x.shape[1], x.shape[2])

    
    
    
    
    
    
    