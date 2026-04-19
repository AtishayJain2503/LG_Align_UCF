"""
generate_embeddings.py
======================
Run this ONCE on the cluster to create:
  embeddings/clip_cvusa_gnd.pt  (ground/street-view CLIP embeddings)
  embeddings/clip_cvusa_sat.pt  (satellite CLIP embeddings)

These files speed up Fahim's per-query evaluation loop by skipping 
the ViT image encoder on every iteration.

Usage (on cluster):
    python generate_embeddings.py

After this script finishes, flip attributes.py:
    use_vis_embed = True
    gnd_embed_pretrn = torch.load("embeddings/clip_cvusa_gnd.pt")
    sat_embed_pretrn = torch.load("embeddings/clip_cvusa_sat.pt")

and run the main eval loop as normal.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast
from transformers import AutoProcessor, AutoTokenizer

from attributes import Configuration as hypm
from custom_models import CLIP_model
from CVUSA_dataset import CVUSA_dataset_cropped

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
DATA_PATH = hypm.data_path        # e.g. /lustre/fs1/home/at387336/LGAlign_project
BATCH_SIZE = hypm.batch_size       # use whatever fits in GPU memory
DEVICE = hypm.device
SAVE_DIR = "embeddings"

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------------------------------------------
# Dataset (val split only - these embeddings are for eval)
# ---------------------------------------------------------------
transform = None   # CLIP processor handles normalisation internally

val_data = pd.read_csv(f'{DATA_PATH}/splits/val-19zl.csv', header=None).reset_index(drop=True)
val_ds   = CVUSA_dataset_cropped(df=val_data, path=DATA_PATH,
                                  transform=transform, train=False,
                                  lang=hypm.lang)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

# ---------------------------------------------------------------
# Model  (backbone only - no fusion head needed)
# ---------------------------------------------------------------
torch.cuda.set_device(hypm.cuda_set_device)
model = CLIP_model(embed_dim=hypm.embed_dim)
model = model.to(DEVICE)
model.eval()

print(f"Generating embeddings over {len(val_ds)} validation pairs...")

gnd_embeds = []
sat_embeds = []

with torch.no_grad():
    for anchor, positive, negative, txt, idx in tqdm(val_loader, desc="Extracting"):
        anchor   = anchor.to(DEVICE)
        positive = positive.to(DEVICE)

        with autocast(device_type='cuda', enabled=hypm.use_mixed_precision):
            # Get raw ViT CLS embeddings (before any MLP head)
            gnd_emb = model.get_vision_embeddings(imgs=anchor,   isQ=True)
            sat_emb = model.get_vision_embeddings(imgs=positive, isQ=False)

        gnd_embeds.append(gnd_emb.cpu())
        sat_embeds.append(sat_emb.cpu())

# ---------------------------------------------------------------
# Concatenate and save
# ---------------------------------------------------------------
gnd_embeds = torch.cat(gnd_embeds, dim=0)
sat_embeds = torch.cat(sat_embeds, dim=0)

print(f"Ground embeddings shape : {gnd_embeds.shape}")
print(f"Satellite embeddings shape : {sat_embeds.shape}")

torch.save(gnd_embeds, f"{SAVE_DIR}/clip_cvusa_gnd.pt")
torch.save(sat_embeds, f"{SAVE_DIR}/clip_cvusa_sat.pt")

print(f"\nSaved to {SAVE_DIR}/clip_cvusa_gnd.pt and {SAVE_DIR}/clip_cvusa_sat.pt")
print("Done! You can now set use_vis_embed = True in attributes.py.")
