import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast

# ---------------------------------------------------------------
# All config inline - no dependency on attributes.py
# ---------------------------------------------------------------
DATA_PATH  = '/lustre/fs1/home/at387336/LGAlign_project'
BATCH_SIZE = 20
LANG       = 'T1'
SAVE_DIR   = 'embeddings'
PRETRAIN   = 'openai/clip-vit-large-patch14'
EMBED_DIM  = 768
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
CUDA_ID    = 0

os.makedirs(SAVE_DIR, exist_ok=True)
torch.cuda.set_device(CUDA_ID)

# ---------------------------------------------------------------
# Import model and dataset after env is set
# ---------------------------------------------------------------
from custom_models import CLIP_model
from CVUSA_dataset import CVUSA_dataset_cropped

# ---------------------------------------------------------------
# Dataset - val split only
# ---------------------------------------------------------------
val_csv  = DATA_PATH + '/splits/val-19zl.csv'
val_data = pd.read_csv(val_csv, header=None).reset_index(drop=True)

val_ds = CVUSA_dataset_cropped(
    df=val_data,
    path=DATA_PATH,
    transform=True,   # processor is applied inside the dataset
    train=False,
    lang=LANG
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# ---------------------------------------------------------------
# Model  (ViT backbone only - MLP head not needed here)
# ---------------------------------------------------------------
model = CLIP_model(embed_dim=EMBED_DIM)
model = model.to(DEVICE)
model.eval()

print("Generating embeddings over " + str(len(val_ds)) + " validation pairs...")

gnd_embeds = []
sat_embeds = []

with torch.no_grad():
    for anchor, positive, negative, txt, idx in tqdm(val_loader, desc="Extracting"):
        anchor   = anchor.to(DEVICE)
        positive = positive.to(DEVICE)

        with autocast(device_type='cuda', enabled=True):
            gnd_emb = model.get_vision_embeddings(imgs=anchor,   isQ=True)
            sat_emb = model.get_vision_embeddings(imgs=positive, isQ=False)

        gnd_embeds.append(gnd_emb.cpu())
        sat_embeds.append(sat_emb.cpu())

# ---------------------------------------------------------------
# Concatenate and save
# ---------------------------------------------------------------
gnd_embeds = torch.cat(gnd_embeds, dim=0)
sat_embeds = torch.cat(sat_embeds, dim=0)

print("Ground embeddings shape    : " + str(gnd_embeds.shape))
print("Satellite embeddings shape : " + str(sat_embeds.shape))

torch.save(gnd_embeds, SAVE_DIR + '/clip_cvusa_gnd.pt')
torch.save(sat_embeds, SAVE_DIR + '/clip_cvusa_sat.pt')

print("\nSaved embeddings to: " + SAVE_DIR)
print("DONE. Now set use_vis_embed = True and uncomment gnd/sat_embed_pretrn in attributes.py")
