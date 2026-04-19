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

# -------------------------------------------Res50---------------------------------------------------------------------------
        # self.vis_embed_shape = 512
        # self.txt_embed_shape = 512

# -------------------------------------------og-EVA---------------------------------------------------------------------------       
        # self.vis_embed_shape = hypm.embed_dim
        # self.txt_embed_shape = self.text.text_projection.out_features
# -------------------------------------------fusion---------------------------------------------------------------------------             
        self.mlp_txt = nn.Linear(self.vis_embed_shape+self.txt_embed_shape, embed_dim ).to(device=self.device)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            batch_first=True
        ).to(device=self.device)
        self.cross_attn_proj = nn.Linear(embed_dim, embed_dim).to(device=self.device)
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






    def get_vision_embeddings(self, imgs, isQ=True):
        if isQ:
            outputs = self.query(imgs)
        else:
            outputs = self.ref(imgs)

        image_embeds = outputs.image_embeds
        return image_embeds

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
        else:
            xlt = xr
        xlt = self.vis_txt_L1(xlt)
        xlt = torch.relu(xlt)
        xlt = self.vis_txt_L2(xlt)
        xlt = torch.relu(xlt)
        xlt = self.vis_txt_L3(xlt)
        return xlt
    
    def get_text_embeddings(self, txt):
        txt = self.tokenizer(txt, padding=True, truncation=True, return_tensors="pt", max_length=77)
        txt = txt.to(device=self.device)
        outputs = self.text(**txt)
        return outputs.text_embeds

    def encode_candidates(self, q, r, t):
        xq = self.get_vision_embeddings(imgs=q, isQ=True)
        xr = self.get_vision_embeddings(imgs=r, isQ=False)
        
        if hypm.use_neg_text:
            xt = self.get_text_embeddings(txt=t[0])
            xt_n = self.get_text_embeddings(txt=t[1])
            return xq, xr, (xt, xt_n)
        else:
            xt = self.get_text_embeddings(txt=t)
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







