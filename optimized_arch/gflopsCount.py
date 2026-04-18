import torch
import torchvision.models as models
from fvcore.nn import FlopCountAnalysis
import torch.nn as nn

from custom_models import CLIP_model
from models.clip_b32 import getClipVisionModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Example model
# model = models.AlexNet()

# model = CLIP_model(embed_dim=768)
# model.eval().to(device=device)
# input1 = torch.randn(1, 3, 224, 224).to(device)

# # flops = FlopCountAnalysis(model, input1)
# # print(f"FLOPs: {flops.total()/ 1e9} GFLOPs")

# # Dummy input
# q = torch.randn(1, 3, 224, 224).to(device)
# r = torch.randn(1, 3, 224, 224).to(device)
# t="sample test text"

# flops = FlopCountAnalysis(model, (q,r,t,True,True))
# print(f"FLOPs: {flops.total()} GFLOPs")
# --------------------------------------------------------------------------
class TwoBranchModel(nn.Module):
    def __init__(self):
        super(TwoBranchModel, self).__init__()
        
        self.mlp_txt = nn.Linear(768*2, 768)
        # Branch 1
        self.branch1 = nn.Sequential(
            nn.Linear(768, 768),
            # nn.ReLU(),
            nn.Linear(768, 768),
            # nn.ReLU(),
            nn.Linear(768, 768)
        )
        
        # Branch 2
        self.branch2 = nn.Sequential(
            nn.Linear(768, 768),
            # nn.ReLU(),
            nn.Linear(768, 768),
            # nn.ReLU(),
            nn.Linear(768, 768)
        )
    
    def forward(self, x1, x2, x3):
        xlt = torch.cat((x2, x3), 1)
        xlt = self.mlp_txt(xlt)
        out1 = self.branch1(x1)
        out2 = self.branch2(xlt)
        return out1, out2


model = TwoBranchModel().to(device=device)
q = torch.randn(1, 768).to(device=device)
r = torch.randn(1, 768).to(device=device)
t = torch.randn(1, 768).to(device=device)


flops = FlopCountAnalysis(model, (q, r, t))
print(f"FLOPs: {flops.total()/ 1e9} GFLOPs")

# -----------------------Param info----------------------------------------
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)


trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable Parameters: {trainable_params:,}")

# ---------------------------------------------------------------
# -------------------------------------------------------------------------------
# model = getClipVisionModel()
# input1 = torch.randn(1, 3, 224, 224).to(device)


# # # Calculate FLOPs
# flops = FlopCountAnalysis(model, (input1, True))
# print(f"FLOPs: {flops.total()/ 1e9:.2f} GFLOPs")