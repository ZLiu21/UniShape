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
from .networks import TokenGeneratorUnit, TransformerEnc, InceptionModule
    

class UniShapeModel(nn.Module):
    def __init__(self, config, series_size: int, in_channels: int, window_emb_dim: int, out_channels: int, shape_ratio: float, scale_len: int,
                 window_size=16, stride=16, depth=6, heads=8, head_dim=16, mlp_dim=512, pool_mode='cls',
                 pe_mode='learnable', max_num_tokens=1024, dropout=0., pos_dropout=0., attn_dropout=0.,
                 path_dropout=0., init_values=None, samples=None, targets=None, pre_training=False, tau=0.5, shape_alpha=0.01):
        super(UniShapeModel, self).__init__()
        
        self.shape_alpha = shape_alpha 
        self.tau = tau
        self.scale_len = scale_len
        
        self.unit_scale_list= nn.ModuleList()
        self.unit_scale_list_finetune = nn.ModuleList()
        self.shape_ratio = shape_ratio
        self.act_gelu_inc = nn.GELU()
        self.layer_norm_inc = nn.LayerNorm(in_channels)
        self.inceptime_token = nn.Sequential(InceptionModule(128, 32), InceptionModule(128, 32))
        self.drop_token = nn.Dropout(p=0.15)
        self.pre_training = pre_training
     
        window_size_list = [64, 32, 16, 8, 4]
        for _shape_size in window_size_list:
            config.window_size = _shape_size
            
            if series_size < 64:
                config.stride = 2
            else:
                config.stride = 4
            
            num_patches = int((series_size - config.window_size) / config.stride + 1)
            
            token_unit = TokenGeneratorUnit(hidden_dim=128,
                                              num_patches=num_patches,
                                              patch_window_size=config.window_size,
                                              scalar_scales=None,
                                              hidden_dim_scalar_enc=32,
                                              epsilon_scalar_enc=1.1,
                                              patch_len=config.window_size,
                                              stide_len=config.stride)
            self.unit_scale_list.append(token_unit)
            
            token_unit_fine = TokenGeneratorUnit(hidden_dim=128,
                                            num_patches=num_patches,
                                            patch_window_size=config.window_size,
                                            scalar_scales=None,
                                            hidden_dim_scalar_enc=32,
                                            epsilon_scalar_enc=1.1,
                                            patch_len=config.window_size,
                                            stide_len=config.stride)
            self.unit_scale_list_finetune.append(token_unit_fine)
        
        self.num_windows = int((series_size - config.window_size) / config.stride) + 1
        num_tokens = self.num_windows + 1 if pool_mode == 'cls' else self.num_windows
        
        self.transformer_enc = TransformerEnc(hidden_dim=in_channels, num_patches=num_tokens, depth=6,
                                heads=8, mlp_dim=512, dim_head=64,
                                dropout=0.1, device=None)
        
        self.act_gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(in_channels * 1)
        
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
    
        window_size_list = [64, 32, 16, 8, 4]
        cls_tokens = None
        x_embed = None
        _i = self.scale_len - 1
        _shape_size = window_size_list[_i]
        
        if _i == 4:
            x_embed = self.unit_scale_list[_i](x)
        else:
            x_embed = self.unit_scale_list_finetune[_i](x)
            
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
        trans_enc_class_token, shape_tokens = self.transformer_enc(x_embed, cls_token_in=cls_tokens)
           
        B, C, D = shape_tokens.shape
        shape_tokens = self.fc_token_shape(shape_tokens.reshape(-1, D)).reshape(B, C, D)
        shape_attn_x_score = self.attention_head(shape_tokens).squeeze() 

        k = int(C * self.shape_ratio)  
        topk_scores, topk_indices = torch.topk(shape_attn_x_score, k=k, dim=1, largest=True, sorted=False) 
    
        topk_indices_exp = topk_indices.unsqueeze(-1).expand(-1, -1, D)  # [B, k, D]
        high_atten_shape_tokens = torch.gather(shape_tokens, dim=1, index=topk_indices_exp)  # [B, k, D]

        cls_results = self.fc(trans_enc_class_token)
        
        features = self.fc_token_shape(trans_enc_class_token)
        
        momentum = 0.9
        C, D = self.class_proto_centers.shape
        batch_centers = torch.zeros_like(self.class_proto_centers)  # (C, D)
        counts = torch.zeros(C, device=features.device)
        for c in labels.unique():
            mask = (labels == c)
            if mask.sum() == 0:
                continue
            class_feat = features[mask].mean(dim=0)
            batch_centers[c] = class_feat
            counts[c] = 1.0

        new_centers = (
            momentum * self.class_proto_centers +
            (1 - momentum) * batch_centers
        )

        new_centers = torch.where(
            counts.view(-1, 1) > 0,
            new_centers,
            self.class_proto_centers
        )

        self.class_proto_centers.data = new_centers.data  
        centers = F.normalize(self.class_proto_centers, dim=1) 
        finetune_ce_loss = F.cross_entropy(cls_results, labels)

        B, T, D = high_atten_shape_tokens.shape
        flattened_tokens = high_atten_shape_tokens.view(B * T, D)
        expanded_labels = labels.unsqueeze(1).expand(-1, T).reshape(-1)  # (B*T,)
        flattened_tokens = F.normalize(flattened_tokens, dim=1)  # (B, D)
        token_proto_logits = torch.matmul(flattened_tokens, centers.T)

        per_token_loss = F.cross_entropy(
            token_proto_logits / self.tau,  
            expanded_labels,
            reduction='none' 
        )

        per_token_loss = per_token_loss.view(B, T)
        weighted_loss = (per_token_loss * topk_scores).sum(dim=1) / topk_scores.sum(dim=1)

        token_proto_loss = weighted_loss.mean()
        
        if self.shape_ratio == 0.0:
            total_loss = finetune_ce_loss
        else:
            total_loss = finetune_ce_loss + self.shape_alpha * token_proto_loss 
        
        return cls_results, total_loss
