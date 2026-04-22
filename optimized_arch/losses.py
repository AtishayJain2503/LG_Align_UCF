import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributed.nn
import math



# # Define Triplet Loss
# class TripletLoss(nn.Module):
#     def __init__(self, margin=0.2):
#         super(TripletLoss, self).__init__()
#         self.margin = margin

#     def forward(self, anchor, positive, negative):
#         distance_positive = torch.sum(torch.pow(anchor - positive, 2), dim=1)
#         distance_negative = torch.sum(torch.pow(anchor - negative, 2), dim=1)
#         losses = torch.relu(distance_positive - distance_negative + self.margin)
#         return torch.mean(losses)


class Contrastive_loss(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.loss_function = loss_function
        self.device = device
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image_features1, image_features2):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        logits_per_image1 = self.logit_scale * image_features1 @ image_features2.T
        
        logits_per_image2 = logits_per_image1.T
        
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        
        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels))/2

        return loss  


class InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.loss_function = loss_function
        self.device = device
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image_features1, image_features2):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        logits_per_image1 = self.logit_scale * image_features1 @ image_features2.T
        
        logits_per_image2 = logits_per_image1.T
        
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        
        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels))/2

        return loss
    

class InfoNCE_2(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCE_2, self).__init__()
        self.temperature = temperature

    def forward(self, query, positive_key, negative_keys_forward, negative_keys_reverse):
        """
        query: (B, D) ground images
        positive_key: (B, D) satellite images
        negative_keys_forward: (B, N, D) negatives for query -> key (Satellites)
        negative_keys_reverse: (B, M, D) negatives for key -> query (Grounds)
        """
        query = F.normalize(query, dim=-1)
        positive_key = F.normalize(positive_key, dim=-1)
        negative_keys_forward = F.normalize(negative_keys_forward, dim=-1)
        negative_keys_reverse = F.normalize(negative_keys_reverse, dim=-1)

        B = query.shape[0]

        # Forward loss: Ground -> Satellite
        pos_sim = F.cosine_similarity(query, positive_key, dim=-1)
        neg_sim_fwd = torch.bmm(negative_keys_forward, query.unsqueeze(-1)).squeeze(-1)
        logits_fwd = torch.cat([pos_sim.unsqueeze(1), neg_sim_fwd], dim=1) / self.temperature
        
        # Reverse loss: Satellite -> Ground
        neg_sim_rev = torch.bmm(negative_keys_reverse, positive_key.unsqueeze(-1)).squeeze(-1)
        logits_rev = torch.cat([pos_sim.unsqueeze(1), neg_sim_rev], dim=1) / self.temperature

        labels = torch.zeros(B, dtype=torch.long, device=query.device)

        loss_fwd = F.cross_entropy(logits_fwd, labels)
        loss_rev = F.cross_entropy(logits_rev, labels)

        return (loss_fwd + loss_rev) / 2



# ─────────────────────────────────────────────────────────────────────────────
# Upgrade #4 — ArcGeo Angular Margin Loss (from ArcGeo, WACV 2024)
# Forces larger angular separation in embedding space — critical for 90° FoV
# where narrow crops create visually ambiguous embeddings.
# Adds a cosine-margin penalty m to the positive pair angle before softmax.
# ─────────────────────────────────────────────────────────────────────────────
class ArcGeoLoss(torch.nn.Module):
    """Batch-all angular margin loss (ArcGeo, WACV 2024).

    For each pair (q_i, r_i), the angular margin m is subtracted from the
    cosine similarity *angle* of the positive pair before computing softmax,
    forcing the decision boundary to maintain a minimum angular gap.

    Args:
        temperature: logit scale denominator (default 0.07)
        margin: additive angular margin in radians (default pi/6 = 30°)
    """
    def __init__(self, temperature: float = 0.07, margin: float = math.pi / 6):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, query, positive_key, negative_keys_forward, negative_keys_reverse):
        """
        query:                  (B, D) — ground embeddings
        positive_key:           (B, D) — satellite embeddings
        negative_keys_forward:  (B, N, D) — hard negative satellites
        negative_keys_reverse:  (B, M, D) — hard negative grounds (may be empty)
        """
        query         = F.normalize(query,         dim=-1)
        positive_key  = F.normalize(positive_key,  dim=-1)
        negative_keys_forward = F.normalize(negative_keys_forward, dim=-1)

        B = query.shape[0]
        labels = torch.zeros(B, dtype=torch.long, device=query.device)

        # ── Forward: Ground → Satellite ───────────────────────────────────────
        cos_pos = F.cosine_similarity(query, positive_key, dim=-1)  # (B,)
        
        # ------------------------------------------------------------------
        # SAFE FP16 ANGULAR MARGIN (ArcFace/ArcGeo trigonometric expansion)
        # cos(theta + margin) = cos(theta)cos(margin) - sin(theta)sin(margin)
        # Avoids torch.acos() which causes NaN gradients in mixed precision!
        # ------------------------------------------------------------------
        cos_m = math.cos(self.margin)
        sin_m = math.sin(self.margin)
        sin_pos = torch.sqrt((1.0 - torch.pow(cos_pos, 2)).clamp(min=1e-6))
        cos_pos_margin = cos_pos * cos_m - sin_pos * sin_m             # (B,)

        neg_sim_fwd = torch.bmm(
            negative_keys_forward, query.unsqueeze(-1)
        ).squeeze(-1)                                               # (B, N)

        logits_fwd = torch.cat(
            [cos_pos_margin.unsqueeze(1), neg_sim_fwd], dim=1
        ) / self.temperature                                        # (B, 1+N)
        loss_fwd = F.cross_entropy(logits_fwd, labels)

        # ── Reverse: Satellite → Ground (skip margin for simplicity) ─────────
        if negative_keys_reverse.shape[1] > 0:
            negative_keys_reverse = F.normalize(negative_keys_reverse, dim=-1)
            neg_sim_rev = torch.bmm(
                negative_keys_reverse, positive_key.unsqueeze(-1)
            ).squeeze(-1)                                           # (B, M)
            logits_rev = torch.cat(
                [cos_pos.unsqueeze(1), neg_sim_rev], dim=1
            ) / self.temperature
            loss_rev = F.cross_entropy(logits_rev, labels)
            return (loss_fwd + loss_rev) / 2

        return loss_fwd


# ─────────────────────────────────────────────────────────────────────────────
# Upgrade #5 — Dynamic Weighted Batch-tuple Loss (VimGeo, IJCAI 2025 / GeoSSM)
# Up-weights hard negatives by their cosine similarity to the query.
# Harder negatives → higher weight → stronger gradient signal.
# Especially effective at 90° FoV where many negatives look similar.
# ─────────────────────────────────────────────────────────────────────────────
class DWBLInfoNCE(torch.nn.Module):
    """InfoNCE with Dynamic Weighted Batch-tuple Loss (VimGeo / GeoSSM, 2025).

    Instead of uniform treatment, each negative's contribution is weighted by
    exp(sim(q, n_i)) so harder negatives receive proportionally more gradient.

    Can be combined with ArcGeoLoss by using this as a secondary objective,
    or used standalone as a drop-in for InfoNCE_2.

    Args:
        temperature: logit scale denominator (default 0.07)
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, query, positive_key, negative_keys_forward, negative_keys_reverse):
        """
        Same signature as InfoNCE_2 for drop-in compatibility.
        """
        query         = F.normalize(query,         dim=-1)  # (B, D)
        positive_key  = F.normalize(positive_key,  dim=-1)  # (B, D)
        negative_keys_forward = F.normalize(negative_keys_forward, dim=-1)  # (B, N, D)

        B = query.shape[0]
        labels = torch.zeros(B, dtype=torch.long, device=query.device)

        # ── Positive similarity ───────────────────────────────────────────────
        pos_sim = F.cosine_similarity(query, positive_key, dim=-1)  # (B,)

        # ── Forward: Ground → Satellite  (dynamic weighting) ─────────────────
        neg_sim_fwd = torch.bmm(
            negative_keys_forward, query.unsqueeze(-1)
        ).squeeze(-1)                                               # (B, N)

        # Dynamic weights: w_i = exp(sim_i) / sum(exp(sim_j))  — softmax over negatives
        neg_weights = torch.softmax(neg_sim_fwd.detach() / self.temperature, dim=-1)  # (B, N)

        logits_fwd = torch.cat(
            [pos_sim.unsqueeze(1), neg_sim_fwd], dim=1
        ) / self.temperature                                        # (B, 1+N)

        # Weighted cross-entropy: scale log-probabilities by dynamic weights
        log_probs_fwd = F.log_softmax(logits_fwd, dim=-1)          # (B, 1+N)
        # Positive is at index 0; negatives at 1..N
        loss_fwd = -log_probs_fwd[:, 0].mean()  # standard term
        # Add weighted penalty for each hard negative exceeding threshold
        loss_fwd = loss_fwd + (neg_weights * (-log_probs_fwd[:, 1:])).sum(dim=-1).mean() * 0.1

        # ── Reverse: Satellite → Ground ───────────────────────────────────────
        if negative_keys_reverse.shape[1] > 0:
            negative_keys_reverse = F.normalize(negative_keys_reverse, dim=-1)
            neg_sim_rev = torch.bmm(
                negative_keys_reverse, positive_key.unsqueeze(-1)
            ).squeeze(-1)
            logits_rev = torch.cat(
                [pos_sim.unsqueeze(1), neg_sim_rev], dim=1
            ) / self.temperature
            loss_rev = F.cross_entropy(logits_rev, labels)
            return (loss_fwd + loss_rev) / 2

        return loss_fwd


# this is equivalent to the loss function in CVMNet with alpha=10, here we simplify it with cosine similarity
class SoftTripletBiLoss(nn.Module):
    def __init__(self, margin=None, alpha=20, **kwargs):
        super(SoftTripletBiLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha

    def forward(self, inputs_q, inputs_k):
        loss_1, mean_pos_sim_1, mean_neg_sim_1 = self.single_forward(inputs_q, inputs_k)
        loss_2, mean_pos_sim_2, mean_neg_sim_2 = self.single_forward(inputs_k, inputs_q)
        return (loss_1+loss_2)*0.5, (mean_pos_sim_1+mean_pos_sim_2)*0.5, (mean_neg_sim_1+mean_neg_sim_2)*0.5

    def single_forward(self, inputs_q, inputs_k):
        n = inputs_q.size(0)
        
        normalized_inputs_q = inputs_q / torch.norm(inputs_q, dim=1, keepdim=True)
        normalized_inputs_k = inputs_k / torch.norm(inputs_k, dim=1, keepdim=True)
        

        # Compute similarity matrix
        sim_mat = torch.matmul(normalized_inputs_q, normalized_inputs_k.t())
        # split the positive and negative pairs
        eyes_ = torch.eye(n).cuda()

        pos_mask = eyes_.eq(1)
        neg_mask = ~pos_mask
        
        # print(pos_mask.shape)

        pos_sim = torch.masked_select(sim_mat, pos_mask)
        neg_sim = torch.masked_select(sim_mat, neg_mask)

        pos_sim_ = pos_sim.unsqueeze(dim=1).expand(n, n-1)
        neg_sim_ = neg_sim.reshape(n, n-1)

        # print(f'after apply unsqueeze{pos_sim_}')
        # print(f'after apply unsqueeze{neg_sim_.shape}')



        loss_batch = torch.log(1 + torch.exp((neg_sim_ - pos_sim_) * self.alpha))
        if torch.isnan(loss_batch).any():
            print(inputs_q, inputs_k)
            raise Exception

        loss = loss_batch.mean()

        mean_pos_sim = pos_sim.mean().item()
        mean_neg_sim = neg_sim.mean().item()
        return loss, mean_pos_sim, mean_neg_sim
    


