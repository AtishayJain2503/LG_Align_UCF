import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
from vit_pytorch import ViT
from models.clip_b32 import getClipTextModel, getClipVisionModel, getClipVisionModelEVA, getTransformerEncoder, getCrossAttention, getClipTextModelRN, getClipVisionModelRN
from transformers import AutoTokenizer, AutoProcessor
import clip

from attributes import Configuration as hypm





# Define the ResNet model
class ResNet(nn.Module):
    def __init__(self, emb_dim = 512):
        super(ResNet, self).__init__()
        self.modelName = 'ResNet18'
        self.q_net = resnet18(ResNet18_Weights.IMAGENET1K_V1)
        self.ref_net = resnet18(ResNet18_Weights.IMAGENET1K_V1)
        # for param in self.q_net.parameters():
        #     param.requires_grad = False
        # for param in self.ref_net.parameters():
        #     param.requires_grad = False
        self.resnet_output = self.q_net.fc.out_features
        # self.fc_q = nn.Linear(self.resnet_output, emb_dim)
        # self.fc_r = nn.Linear(self.resnet_output, emb_dim)
        # self.sigmoid = nn.Sigmoid()





    def forward(self, q, r, isTrain = True, isQuery = True):
        xq = self.q_net(q)
        # xq = self.fc_q(xq)
        # xq = torch.sigmoid(xq)

        xr = self.ref_net(r)
        # xr = self.fc_r(xr)
        # xr = torch.sigmoid(xr)
        
        if isTrain:
            # print(f'dukse train')
            return xq, xr
            # return self.query.encode_image(q), self.ref.encode_image(r)
        else:
            if isQuery:
                # print(f'dukse query')
                return xq
                # return self.query.encode_image(q)
            else:
                # print(f'dukse ref')
                return xr
                # return self.ref.encode_image(r)

    



class ResNet2(nn.Module):
    def __init__(self, emb_dim):
        super(ResNet2, self).__init__()
        self.modelName = 'ResNet18'
        self.net = resnet18()



    def forward(self, img):
        return self.net(img)


# Define the VIT model
class VIT(nn.Module):
    def __init__(self):
        super(VIT, self).__init__()
        self.modelName = 'VIT'
        self.query = ViT(
            image_size = 256,
            patch_size = 32,
            num_classes = 512,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
            )
        self.ref = ViT(
            image_size = 256,
            patch_size = 32,
            num_classes = 512,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
            )



    def forward(self, q, r, isTrain = True, isQuery = True):
        if isTrain:
            return self.query(q), self.ref(r)
        else:
            if isQuery:
                return self.query(q)
            else:
                return self.ref(r)




class LiFtQFormer(nn.Module):
    def __init__(self, embed_dim=512, num_queries=4, num_layers=2):
        super().__init__()
        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.out_proj = nn.Linear(embed_dim * num_queries, embed_dim)

    def forward(self, v_emb, t_emb):
        B = v_emb.shape[0]
        # Treat the two modal embeddings as a sequence of length 2
        kv = torch.stack([v_emb, t_emb], dim=1) # [B, 2, embed_dim]
        
        queries = self.query_tokens.expand(B, -1, -1) # [B, num_queries, embed_dim]
        
        out = self.transformer(tgt=queries, memory=kv) # [B, num_queries, embed_dim]
        
        out = out.reshape(B, -1) # [B, num_queries * embed_dim]
        out = self.out_proj(out) # [B, embed_dim]
        return out


class LiFtQFormerSpatial(nn.Module):
    def __init__(self, embed_dim=768, hidden_size=1024, num_queries=4, num_layers=2):
        super().__init__()
        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        
        self.vis_proj = nn.Linear(hidden_size, embed_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.out_proj = nn.Linear(embed_dim * num_queries, embed_dim)

    def forward(self, vis_hidden, txt_emb):
        B = vis_hidden.shape[0]
        # vis_hidden: [B, seq_v, hidden_size]
        # txt_emb: [B, embed_dim]
        
        v_proj = self.vis_proj(vis_hidden) # [B, seq_v, embed_dim]
        t_proj = txt_emb.unsqueeze(1)      # [B, 1, embed_dim]
        
        kv = torch.cat([v_proj, t_proj], dim=1) # [B, seq_v + 1, embed_dim]
        
        queries = self.query_tokens.expand(B, -1, -1) # [B, num_queries, embed_dim]
        
        out = self.transformer(tgt=queries, memory=kv) # [B, num_queries, embed_dim]
        
        out = out.reshape(B, -1) # [B, num_queries * embed_dim]
        out = self.out_proj(out) # [B, embed_dim]
        return out

class FlamingoGatedCrossAttention(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        # Cross attention where Satellite image queries the Text sequence
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            batch_first=True
        )
        # The key Flamingo innovation: start gate at 0 so it defaults to pure visual
        self.gate = nn.Parameter(torch.zeros(1))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, v_emb, t_seq):
        # v_emb: [B, D] (satellite pooled embedding)
        # t_seq: [B, N, D] (text token sequence)
        
        # We need the query to be 3D: [B, 1, D]
        q = v_emb.unsqueeze(1)
        
        # Satellite queries the Text sequence
        attn_out, _ = self.cross_attn(query=q, key=t_seq, value=t_seq)
        attn_out = attn_out.squeeze(1) # [B, D]
        
        attn_out = self.out_proj(attn_out)
        
        # Residual gating
        xlt = v_emb + torch.tanh(self.gate) * attn_out
        return xlt

# Define the Hugging face CLIP model
class CLIP_model(nn.Module):
    def __init__(self, embed_dim):
        super(CLIP_model, self).__init__()
        self.modelName = 'CLIP'
        self.device = hypm.device
        self.tokenizer = AutoTokenizer.from_pretrained(hypm.t_pretrain_weight)
        self.processor = AutoProcessor.from_pretrained(hypm.v_pretrain_weight)


#-----------------------OG-ViT--------------------------
        self.query = getClipVisionModel()
        # Phase 1: Freeze backbone — only MLP heads train at high LR initially
        for param in self.query.parameters():
            param.requires_grad = False

        # Separate encoder for satellite — different domain, independent fine-tuning
        self.ref = getClipVisionModel()
        for param in self.ref.parameters():
            param.requires_grad = False

        self.text = getClipTextModel()
        # for param in self.text.parameters():
        #     param.requires_grad = False

#-----------------------Res50------------------------
        # self.query = getClipVisionModelRN()
        # self.ref = getClipVisionModelRN()
        # self.text = getClipTextModelRN()

#-----------------------OG-EVA--------------------------
        # self.query = getClipVisionModelEVA()
        # # for param in self.query.parameters():
        # #     param.requires_grad = False

        # self.ref = getClipVisionModelEVA()
        # # for param in self.ref.parameters():
        # #     param.requires_grad = False

        # self.text = getClipTextModel()
        # # for param in self.text.parameters():
        # #     param.requires_grad = False
#------------------------------------------------------



        # self.norm_shape = self.query.vision_model.post_layernorm.normalized_shape[0]
        

# -------------------------------------------og---------------------------------------------------------------------------       
        self.vis_embed_shape = self.query.visual_projection.out_features
        self.txt_embed_shape = self.text.text_projection.out_features
        self.vis_hidden_shape = self.query.vision_model.post_layernorm.normalized_shape[0]

# -------------------------------------------Res50---------------------------------------------------------------------------
        # self.vis_embed_shape = 512
        # self.txt_embed_shape = 512

# -------------------------------------------og-EVA---------------------------------------------------------------------------       
        # self.vis_embed_shape = hypm.embed_dim
        # self.txt_embed_shape = self.text.text_projection.out_features
# -------------------------------------------fusion---------------------------------------------------------------------------             
        self.mlp_txt = nn.Linear(self.vis_embed_shape+self.txt_embed_shape, embed_dim ).to(device=self.device)

        self.qformer = LiFtQFormer(embed_dim=self.vis_embed_shape, num_queries=4, num_layers=2).to(device=self.device)
        self.qformer_spatial = LiFtQFormerSpatial(embed_dim=self.vis_embed_shape, hidden_size=self.vis_hidden_shape, num_queries=4, num_layers=2).to(device=self.device)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            batch_first=True
        ).to(device=self.device)
        self.cross_attn_proj = nn.Linear(embed_dim, embed_dim).to(device=self.device)
        
        self.flamingo = FlamingoGatedCrossAttention(embed_dim=self.vis_embed_shape).to(device=self.device)
        
        # V10b: Independent linear projections (professor's suggestion)
        # xlt = W_sat @ xr + W_txt @ xt  — fully decoupled, no non-linearity in fusion
        self.linear_proj_sat = nn.Linear(self.vis_embed_shape, embed_dim).to(device=self.device)
        self.linear_proj_txt = nn.Linear(self.txt_embed_shape, embed_dim).to(device=self.device)
        # self.vis_gnd_L1 = nn.Linear(embed_dim, embed_dim).to(device=self.device)
        # self.txt_gnd_L1 = nn.Linear(embed_dim, embed_dim).to(device=self.device)



        self.vis_txt_L1 = nn.Linear(embed_dim, embed_dim).to(device=self.device)
        self.vis_txt_L2 = nn.Linear(embed_dim, embed_dim).to(device=self.device)
        self.vis_txt_L3 = nn.Linear(embed_dim, embed_dim).to(device=self.device)
        # self.vis_txt_L4 = nn.Linear(embed_dim, embed_dim).to(device=self.device)
        # self.vis_txt_L5 = nn.Linear(embed_dim, embed_dim).to(device=self.device)
        # self.vis_txt_L6 = nn.Linear(embed_dim, embed_dim).to(device=self.device)



        self.vis_L1 = nn.Linear(embed_dim, embed_dim).to(device=self.device)
        self.vis_L2 = nn.Linear(embed_dim, embed_dim).to(device=self.device)
        self.vis_L3 = nn.Linear(embed_dim, embed_dim).to(device=self.device)
        # self.vis_L4 = nn.Linear(embed_dim, embed_dim).to(device=self.device)
        # self.vis_L5 = nn.Linear(embed_dim, embed_dim).to(device=self.device)
        # self.vis_L6 = nn.Linear(embed_dim, embed_dim).to(device=self.device)

# ----------------------------------------------patch------------------------------------------------------------------------
        # self.patch_temp_q = nn.Linear(1792, embed_dim).to(device=self.device) #patch
        # self.patch_temp_r = nn.Linear(1792, embed_dim).to(device=self.device) #patch


# ----------------------------------------------Cross Attention------------------------------------------------------------------------
        
        # self.mlp_txt = getCrossAttention(dim_in=self.vis_embed_shape, d_out_v=self.vis_embed_shape).to(device=self.device) #cross-atten
# ----------------------------------------------encoder------------------------------------------------------------------------

        # self.trnfr_encoder_1 = torch.nn.TransformerEncoderLayer(embed_dim, 3, 2024, 0.1).to(device=self.device)
        # self.trnfr_encoder_2 = torch.nn.TransformerEncoderLayer(embed_dim, 3, 2024, 0.1).to(device=self.device)

        # self.trnfr_encoder_1 = getTransformerEncoder(dim_in = embed_dim).to(device=self.device)
        # self.trnfr_encoder_2 = getTransformerEncoder(dim_in = embed_dim).to(device=self.device)






    def get_vision_embeddings(self, imgs, isQ=True, return_seq=False):
        if isQ:
            outputs = self.query(imgs, output_hidden_states=return_seq)
        else:
            outputs = self.ref(imgs, output_hidden_states=return_seq)

        if return_seq:
            return outputs.image_embeds, outputs.hidden_states[-1]
        return outputs.image_embeds

    def project_query(self, xq):
        """Project ground-image embeddings through the query MLP head."""
        xq = self.vis_L1(xq)
        xq = torch.relu(xq)
        xq = self.vis_L2(xq)
        xq = torch.relu(xq)
        xq = self.vis_L3(xq)
        return xq

    def fuse_satellite(self, xr, xt=None):
        """Fuse satellite visual + text embeddings through the satellite MLP head.
        Each satellite is paired with its OWN text (not the query's text).
        """
        if hypm.fusion_mode == 'mlp' and xt is not None:
            xlt = torch.cat((xr, xt), dim=1)   # [B, vis+txt]
            xlt = self.mlp_txt(xlt)             # [B, D]
        elif hypm.fusion_mode == 'qformer' and xt is not None:
            xlt = self.qformer(xr, xt)
        elif hypm.fusion_mode == 'qformer_patch' and xt is not None:
            xr_pooled, xr_seq = xr
            xlt = self.qformer_spatial(xr_seq, xt)
        elif hypm.fusion_mode == 'flamingo' and xt is not None:
            xt_pooled, xt_seq = xt
            xlt = self.flamingo(xr, xt_seq)
        elif hypm.fusion_mode == 'linear' and xt is not None:
            xlt = self.linear_proj_sat(xr) + self.linear_proj_txt(xt)
        else:
            if isinstance(xr, tuple):
                xr = xr[0]
            xlt = xr
        xlt = self.vis_txt_L1(xlt)
        xlt = torch.relu(xlt)
        xlt = self.vis_txt_L2(xlt)
        xlt = torch.relu(xlt)
        xlt = self.vis_txt_L3(xlt)
        return xlt
    
    def get_text_embeddings(self, txt, return_seq=False):
        txt = self.tokenizer(txt, padding=True, truncation=True, return_tensors="pt", max_length=77)
        txt = txt.to(device=self.device)
        outputs = self.text(**txt)
        if return_seq:
            return outputs.text_embeds, outputs.last_hidden_state
        return outputs.text_embeds

    def encode_candidates(self, q, r, t):
        xq = self.get_vision_embeddings(imgs=q, isQ=True)
        
        return_seq = (hypm.fusion_mode == 'qformer_patch')
        if return_seq:
            xr_pooled, xr_seq = self.get_vision_embeddings(imgs=r, isQ=False, return_seq=True)
            xr = (xr_pooled, xr_seq)
        else:
            xr = self.get_vision_embeddings(imgs=r, isQ=False)
        
        return_txt_seq = (hypm.fusion_mode == 'flamingo')
        if hypm.use_neg_text:
            xt = self.get_text_embeddings(txt=t[0], return_seq=return_txt_seq)
            xt_n = self.get_text_embeddings(txt=t[1], return_seq=return_txt_seq)
            return xq, xr, (xt, xt_n)
        else:
            xt = self.get_text_embeddings(txt=t, return_seq=return_txt_seq)
            return xq, xr, xt
            
    def fuse_and_project(self, xq, xr, xt):
        if hypm.fusion_mode == 'mlp':        # A4 — MLP concat
            xlt = torch.cat((xr, xt), dim=1)
            xlt = self.mlp_txt(xlt)
            xlt = self.vis_txt_L1(xlt)
            xlt = torch.relu(xlt)
            xlt = self.vis_txt_L2(xlt)
            xlt = torch.relu(xlt)
            xlt = self.vis_txt_L3(xlt)
            
            xq_proj = self.vis_L1(xq)
            xq_proj = torch.relu(xq_proj)
            xq_proj = self.vis_L2(xq_proj)
            xq_proj = torch.relu(xq_proj)
            xq_proj = self.vis_L3(xq_proj)
            return xq_proj, xlt, -1

        elif hypm.fusion_mode == 'qformer':  # LiFt-Q (Q-Former) bottleneck
            xlt = self.qformer(xr, xt)
            xlt = self.vis_txt_L1(xlt)
            xlt = torch.relu(xlt)
            xlt = self.vis_txt_L2(xlt)
            xlt = torch.relu(xlt)
            xlt = self.vis_txt_L3(xlt)
            
            xq_proj = self.vis_L1(xq)
            xq_proj = torch.relu(xq_proj)
            xq_proj = self.vis_L2(xq_proj)
            xq_proj = torch.relu(xq_proj)
            xq_proj = self.vis_L3(xq_proj)
            return xq_proj, xlt, -1

        elif hypm.fusion_mode == 'qformer_patch':  # Spatial Q-Former bottleneck
            xr_pooled, xr_seq = xr
            xlt = self.qformer_spatial(xr_seq, xt)
            xlt = self.vis_txt_L1(xlt)
            xlt = torch.relu(xlt)
            xlt = self.vis_txt_L2(xlt)
            xlt = torch.relu(xlt)
            xlt = self.vis_txt_L3(xlt)
            
            xq_proj = self.vis_L1(xq)
            xq_proj = torch.relu(xq_proj)
            xq_proj = self.vis_L2(xq_proj)
            xq_proj = torch.relu(xq_proj)
            xq_proj = self.vis_L3(xq_proj)
            return xq_proj, xlt, -1
            
        elif hypm.fusion_mode == 'flamingo':       # Flamingo Gated Cross-Attention
            xt_pooled, xt_seq = xt
            xlt = self.flamingo(xr, xt_seq)
            xlt = self.vis_txt_L1(xlt)
            xlt = torch.relu(xlt)
            xlt = self.vis_txt_L2(xlt)
            xlt = torch.relu(xlt)
            xlt = self.vis_txt_L3(xlt)
            
            xq_proj = self.vis_L1(xq)
            xq_proj = torch.relu(xq_proj)
            xq_proj = self.vis_L2(xq_proj)
            xq_proj = torch.relu(xq_proj)
            xq_proj = self.vis_L3(xq_proj)
            return xq_proj, xlt, -1

        elif hypm.fusion_mode == 'linear':          # V10b — Independent Linear Projections
            xlt = self.linear_proj_sat(xr) + self.linear_proj_txt(xt)
            xlt = self.vis_txt_L1(xlt)
            xlt = torch.relu(xlt)
            xlt = self.vis_txt_L2(xlt)
            xlt = torch.relu(xlt)
            xlt = self.vis_txt_L3(xlt)
            
            xq_proj = self.vis_L1(xq)
            xq_proj = torch.relu(xq_proj)
            xq_proj = self.vis_L2(xq_proj)
            xq_proj = torch.relu(xq_proj)
            xq_proj = self.vis_L3(xq_proj)
            return xq_proj, xlt, -1

        elif hypm.fusion_mode == 'cross_attn':   # A5 — cross-attention
            raise NotImplementedError('Patch-level cross-attention is not yet implemented.')

        else:                                    # A3 — vision only
            xr_proj = self.vis_txt_L1(xr)
            xr_proj = torch.relu(xr_proj)
            xr_proj = self.vis_txt_L2(xr_proj)
            xr_proj = torch.relu(xr_proj)
            xr_proj = self.vis_txt_L3(xr_proj)
            
            xq_proj = self.vis_L1(xq)
            xq_proj = torch.relu(xq_proj)
            xq_proj = self.vis_L2(xq_proj)
            xq_proj = torch.relu(xq_proj)
            xq_proj = self.vis_L3(xq_proj)
            return xq_proj, xr_proj, -1

    def forward(self, q, r, t, isTrain=True, isQuery=True):
        xq, xr, xt_pack = self.encode_candidates(q, r, t)
        
        if hypm.use_neg_text:
            xt = xt_pack[0]
            xt_n = xt_pack[1]
        else:
            xt = xt_pack
            
        return self.fuse_and_project(xq, xr, xt)








# Define the ResNet model
# class VIT(nn.Module):
#     def __init__(self):
#         super(VIT, self).__init__()
#         self.modelName = 'VIT_B_16'
#         self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
#         # for param in self.vit.parameters():
#         #     param.requires_grad = False
#         # num_features = self.vit.heads
#         # self.vit.fc = nn.Linear(num_features, emb_dim)



#     def forward(self, x):
#         return self.vit(x)


# # Define the ResNet model
# class ResNet(nn.Module):
#     def __init__(self, emb_dim):
#         super(ResNet, self).__init__()
#         self.resnet = torchvision.models.resnet50(pretrained=True)
#         num_features = self.resnet.fc.in_features
#         self.resnet.fc = nn.Linear(num_features, emb_dim)

#     def forward(self, x):
#         return self.resnet(x)







