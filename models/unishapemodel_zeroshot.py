import argparse
import datetime
import os

import numpy as np
import torch
import math
import torch.nn as nn
from torch.distributions.normal import Normal
from timm.models.layers import trunc_normal_
import torch.nn.functional as F

from .networks import TokenGeneratorUnit, InceptionModule, TransformerEnc


class UniShapeModel(nn.Module):  
    def __init__(self, config, series_size: int, in_channels: int, window_emb_dim: int, out_channels: int,
                 window_size=16, stride=16, depth=6, heads=8, head_dim=16, mlp_dim=512, pool_mode='cls',
                 pe_mode='learnable', max_num_tokens=1024, dropout=0., pos_dropout=0., attn_dropout=0.,
                 path_dropout=0., init_values=None, samples=None, targets=None, pre_training=False, shape_ratio=0.3, proto_tau=0.5, shape_alpha=0.01, shape_sparse_ratio=0.4, scale_len=4):
        super(UniShapeModel, self).__init__()
        
        self.shape_sparse_ratio = shape_sparse_ratio
        self.shape_alpha = shape_alpha 
        self.proto_tau = proto_tau
        self.scale_len = scale_len
        self.encoder_scale_list= nn.ModuleList()
        self.shape_ratio = shape_ratio
        self.act_gelu_inc = nn.GELU()
        self.layer_norm_inc = nn.LayerNorm(in_channels)
        self.inceptime_token = nn.Sequential(InceptionModule(128, 32), InceptionModule(128, 32))
        self.drop_token = nn.Dropout(p=0.15)
     
        window_size_list = [64, 32, 16, 8]
        for _shape_size in window_size_list:
            config.window_size = _shape_size
            
            if series_size < 64:
                config.stride = 2
            else:
                config.stride = 4
                
            if pre_training:
                config.stride = config.window_size
            
            num_patches = int((series_size - config.window_size) / config.stride + 1)
            
            tokgen_unit = TokenGeneratorUnit(hidden_dim=128,
                                              num_patches=num_patches,
                                              patch_window_size=config.window_size,
                                              scalar_scales=None,
                                              hidden_dim_scalar_enc=32,
                                              epsilon_scalar_enc=1.1,
                                              patch_len=config.window_size,
                                              stide_len=config.stride)
            
            self.encoder_scale_list.append(tokgen_unit)
       
        self.num_windows = int((series_size - config.window_size) / config.stride) + 1
        num_tokens = self.num_windows + 1 if pool_mode == 'cls' else self.num_windows
        
        self.pre_training = pre_training
        self.vit_unit = TransformerEnc(hidden_dim=in_channels, num_patches=num_tokens, depth=6,
                                heads=8, mlp_dim=512, dim_head=64,
                                dropout=0.1, device=None)
        
        self.act_gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(in_channels * 1)
        
        if self.pre_training:
            self.fc = nn.Linear(in_channels * 1, out_channels)
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_channels, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, out_channels)
            )
        
        self.fc_token_shape = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, in_channels)
        )
        
        self.class_proto_centers = nn.Parameter(torch.zeros(out_channels, in_channels))
        self.attention_head = nn.Sequential(
            nn.Linear(128, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        # init weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: 
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, labels=None):
        B, C, L = x.shape
        N = self.num_windows
    
        window_size_list = [64, 32, 16, 8]
        cls_tokens = None
        x_embed = None
        for _i in range(self.scale_len):
            _shape_size = window_size_list[_i]
            if (L - 4) <= _shape_size:
                continue
            
            x_embed = self.encoder_scale_list[_i](x)
            if cls_tokens == None:
                cls_incep_token_list = self.inceptime_token(x_embed.permute(0,2,1)).permute(0,2,1)
            else:
                input_x_embed = torch.cat((cls_tokens, x_embed), dim=1).permute(0,2,1)
                cls_incep_token_list = self.inceptime_token(input_x_embed).permute(0,2,1)
                
            cls_incep_token_list = self.drop_token(self.layer_norm_inc(cls_incep_token_list))
            cls_incep_token_list = self.act_gelu_inc(cls_incep_token_list)
            _attn_x_score = self.attention_head(cls_incep_token_list)
            attn_shape_embds = cls_incep_token_list * _attn_x_score
            
            cls_tokens = torch.mean(attn_shape_embds, dim=1).unsqueeze(1)       
            
        x_embed = x_embed.squeeze(1)
        trans_enc_class_token, _ = self.vit_unit(x_embed, cls_token_in=cls_tokens)
    
        return trans_enc_class_token