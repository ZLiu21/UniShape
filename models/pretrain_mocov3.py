import copy
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from utils.util import concat_all_gather, is_dist_avail_and_initialized


def generate_pseudo_labels_by_similarity(features, labels, batch_mask, class_centers, top_ratio=0.7):
    with torch.no_grad():
        # Normalize features and class centers
        normed_feat = F.normalize(features, p=2, dim=1)  # (B, D)
        normed_centers = F.normalize(class_centers, p=2, dim=1)  # (C, D)

        # Compute cosine similarities
        cos_sims = torch.matmul(normed_feat, normed_centers.T)  # (B, C)
        max_sims, pseudo_labels = cos_sims.max(dim=1)  # (B,)
        
        # Mask for unlabeled data
        unlabeled_mask = ~batch_mask

        # Select confident pseudo-labels based on top_ratio
        if top_ratio is not None:
            unlabeled_scores = max_sims[unlabeled_mask]
            num_top = max(1, int(top_ratio * unlabeled_scores.size(0)))
            threshold = torch.topk(unlabeled_scores, num_top, sorted=True).values[-1]
            confident_mask = (max_sims >= threshold) & unlabeled_mask

        # Initialize final targets with pseudo-labels
        final_targets = pseudo_labels.clone()
        final_targets[batch_mask] = labels[batch_mask]  # Replace with real labels

        # Combine confident pseudo-labels with true labels
        train_mask = batch_mask | confident_mask

    return final_targets, train_mask


class MoCoV3ModelPseudo(nn.Module):
    def __init__(self, backbone: nn.Module, lamb, num_classes, shape_ratio, tau=0.1, proto_tau=0.5):
        super(MoCoV3ModelPseudo, self).__init__()
        self.tau = tau
        self.proto_tau = proto_tau
        self.shape_ratio = shape_ratio
        self.lamb = lamb
        self.backbone = backbone
        self.momentum_backbone = copy.deepcopy(backbone)
        prev_dim = self.backbone.fc.in_features
        out_dim = 128
        mlp_dim = out_dim * 2
        
        self.class_centers = nn.Parameter(torch.zeros(num_classes, prev_dim))

        del self.backbone.fc, self.momentum_backbone.fc
        self.backbone.fc = self._build_mlp(3, prev_dim, mlp_dim, out_dim)
        self.momentum_backbone.fc = self._build_mlp(3, prev_dim, mlp_dim, out_dim)
        self.predictor = self._build_mlp(2, out_dim, mlp_dim, out_dim)

        for param, param_m in zip(self.backbone.parameters(), self.momentum_backbone.parameters()):
            param_m.data.copy_(param.data)
            param_m.requires_grad = False

    def forward(self, x1, x2, labels=None, momentum=0.999, train_epoch=10, batch_mask=None):
      
        features_q1, shape_tokens_q1, shape_attn_x_score_q1, vit_out_score_q1 = self.backbone(x1)  # (2B, D)
        features_q2, shape_tokens_q2, shape_attn_x_score_q2, vit_out_score_q2 = self.backbone(x2)  # (2B, D)
        
        q1 = self.predictor(features_q1)
        q2 = self.predictor(features_q2)

        with torch.no_grad():
            self._update_momentum_encoder(momentum)
            
            k1, shape_tokens_k1, shape_attn_x_score_k1, vit_out_score_k1 = self.backbone(x1)  # (2B, D)
            k2, shape_tokens_k2, shape_attn_x_score_k2, vit_out_score_k2 = self.backbone(x2)  # (2B, D)
            
        # similarity scores
        logits1, labels1 = self._compute_logits(q1, k2)
        logits2, labels2 = self._compute_logits(q2, k1)
        
        unsimclr_loss = (nn.CrossEntropyLoss()(logits1, labels1) + nn.CrossEntropyLoss()(logits2, labels2)) * (2 * self.tau)
        
        features = torch.cat([q1, k2, q2, k1], dim=0)  # [2B, D]
        labels = labels.repeat(4)   # [2B]
        
        batch_mask = batch_mask.repeat(4)
        
        shape_tokens = torch.cat([shape_tokens_q1, shape_tokens_k2, 
                                   shape_tokens_q2, shape_tokens_k1], dim=0)                   
        _, L, D = shape_tokens.shape
        k = int(L * self.shape_ratio)  
        shape_attn_x_score = torch.cat([shape_attn_x_score_q1, shape_attn_x_score_k2, shape_attn_x_score_q2, shape_attn_x_score_k1], dim=0)

        attn_scores = shape_attn_x_score.squeeze(-1)
        shape_tokens = shape_tokens * shape_attn_x_score

        topk_scores, topk_indices = torch.topk(attn_scores, k=k, dim=1, largest=True, sorted=False)  # [B, k]
        topk_indices_exp = topk_indices.unsqueeze(-1).expand(-1, -1, D)  # [B, k, D]
        high_atten_shape_tokens = torch.gather(shape_tokens, dim=1, index=topk_indices_exp)  # [B, k, D]
        
        _pro_features = features[batch_mask]
        _pro_labels = labels[batch_mask]
        # Momentum update of class centers
        with torch.no_grad():
            momentum = 0.9
            unique_labels = labels.unique()
            new_centers = self.class_centers.clone()

            for c in unique_labels:
                mask = (_pro_labels == c)
                if mask.sum() == 0:
                    continue
                class_feat = _pro_features[mask].mean(dim=0)  
                new_centers[c] = momentum * self.class_centers[c] + (1 - momentum) * class_feat

            self.class_centers.index_copy_(0, unique_labels, new_centers[unique_labels])
            self.class_centers = F.normalize(self.class_centers, p=2, dim=1)  # (C, D)
            
        if train_epoch > -1:  ## 10
            if len(labels[batch_mask]) > 0:
           
                final_targets, train_mask = generate_pseudo_labels_by_similarity(
                    features=features,
                    labels=labels,
                    batch_mask=batch_mask,
                    class_centers=self.class_centers
                )

                proto_logits = torch.matmul(features, self.class_centers.T)  # (2B, C)
                proto_loss = F.cross_entropy(proto_logits[train_mask] / self.proto_tau, final_targets[train_mask])
                
                high_atten_shape_tokens = high_atten_shape_tokens[train_mask]

                B, T, D = high_atten_shape_tokens.shape
                flattened_tokens = high_atten_shape_tokens.view(B * T, D)
                expanded_labels = final_targets[train_mask].unsqueeze(1).expand(-1, T).reshape(-1)  # (B*T,)
                token_proto_logits = torch.matmul(flattened_tokens, self.class_centers.T)
                token_proto_loss = F.cross_entropy(token_proto_logits / self.proto_tau, expanded_labels)
                
                total_proto_loss = (1 - self.lamb) * proto_loss + self.lamb * token_proto_loss  # 或使用 α 加权

                total_loss = unsimclr_loss + total_proto_loss
            else:
                total_loss = unsimclr_loss
        else:
            proto_loss = 0.0 
            total_loss = unsimclr_loss
        
        return total_loss, logits1, labels1, logits2, labels2

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        for param, param_m in zip(self.backbone.parameters(), self.momentum_backbone.parameters()):
            param_m.data = param_m.data * m + param.data * (1. - m)

    def _compute_logits(self, q, k):
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        k = concat_all_gather(k)
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.tau
        B = logits.shape[0]  # batch size per GPU
        
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0  

        labels = torch.arange(B, dtype=torch.long, device=q.device) + B * rank

        return logits, labels 

    def supervised_contrastive_loss(self, logits, labels_q, labels_k):
        """
        logits: [B, B_total], similarities between q and gathered k
        labels_q: [B]
        labels_k: [B_total]
        """
        mask = torch.eq(labels_q.view(-1, 1), labels_k.view(1, -1)).float().to(logits.device)

        logits = logits - logits.max(dim=1, keepdim=True)[0]
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob_pos.mean()
        return loss

    def _build_mlp(self, num_layers, in_dim, mlp_dim, out_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = in_dim if l == 0 else mlp_dim
            dim2 = out_dim if l == num_layers - 1 else mlp_dim
            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers-1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True)) ##
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)


@torch.no_grad()
def concat_all_gather(tensor):
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    tensors_gather = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)

