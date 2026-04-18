import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributed.nn



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

    def forward(self, query, positive_key, negative_keys):
        """
        query: Tensor of shape (B, D) - batch of queries
        positive_key: Tensor of shape (B, D) - batch of positive keys (true matches)
        negative_keys: Tensor of shape (B, N, D) - batch of negative keys (distractors)
        """
        query = F.normalize(query, dim=-1)
        positive_key = F.normalize(positive_key, dim=-1)
        negative_keys = F.normalize(negative_keys, dim=-1)

        B, D = query.shape  # Batch size, embedding dimension
        N = negative_keys.shape[1]  # Number of negatives

        # Compute similarity between query and positive key
        pos_sim = F.cosine_similarity(query, positive_key, dim=-1)  # (B,)
        
        # Compute similarity between query and all negative keys
        neg_sim = torch.bmm(negative_keys, query.unsqueeze(-1)).squeeze(-1)  # (B, N, D)X(B, D, 1) = (B, N)

        # Stack positive similarity with negatives
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, N+1)
        logits /= self.temperature  # Scale by temperature

        # Labels (0 because the first entry is always the positive)
        labels = torch.zeros(B, dtype=torch.long, device=query.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss  



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
    


