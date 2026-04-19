import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.amp import autocast
from PIL import Image
from transformers import AutoProcessor

# ---------------------------------------------------------------
# All config inline - NO dependency on attributes.py
# ---------------------------------------------------------------
DATA_PATH  = '/lustre/fs1/home/at387336/LGAlign_project'
BATCH_SIZE = 20
SAVE_DIR   = 'embeddings'
PRETRAIN   = 'openai/clip-vit-large-patch14'
EMBED_DIM  = 768
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
CUDA_ID    = 0

os.makedirs(SAVE_DIR, exist_ok=True)
torch.cuda.set_device(CUDA_ID)

# ---------------------------------------------------------------
# Minimal image-only Dataset - no lang CSV needed at all
# ---------------------------------------------------------------
class ImageOnlyDataset(Dataset):
    def __init__(self, csv_path, data_path, processor):
        self.df        = pd.read_csv(csv_path, header=None)
        self.sat_paths = self.df.iloc[:, 0].values   # satellite (bingmap)
        self.gnd_paths = self.df.iloc[:, 1].values   # ground (streetview)
        self.data_path = data_path
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        gnd_img = Image.open(self.data_path + '/' + self.gnd_paths[idx]).convert('RGB')
        sat_img = Image.open(self.data_path + '/' + self.sat_paths[idx]).convert('RGB')

        gnd = self.processor(images=gnd_img, return_tensors='pt')['pixel_values'].squeeze(0)
        sat = self.processor(images=sat_img, return_tensors='pt')['pixel_values'].squeeze(0)
        return gnd, sat

# ---------------------------------------------------------------
# Load processor and dataset
# ---------------------------------------------------------------
print("Loading CLIP processor...")
processor = AutoProcessor.from_pretrained(PRETRAIN)

val_csv = DATA_PATH + '/splits/val-19zl.csv'
dataset = ImageOnlyDataset(csv_path=val_csv, data_path=DATA_PATH, processor=processor)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                     num_workers=4, pin_memory=True)

print("Dataset size: " + str(len(dataset)) + " pairs")

# ---------------------------------------------------------------
# Load model
# ---------------------------------------------------------------
from custom_models import CLIP_model
from attributes import Configuration as hypm

model = CLIP_model(embed_dim=EMBED_DIM)
model = model.to(DEVICE)
model.eval()

print("Extracting embeddings...")

gnd_embeds = []
sat_embeds = []

with torch.no_grad():
    for gnd, sat in tqdm(loader, desc="Batches"):
        gnd = gnd.to(DEVICE)
        sat = sat.to(DEVICE)

        with autocast(device_type='cuda', enabled=True):
            gnd_emb = model.get_vision_embeddings(imgs=gnd, isQ=True)
            sat_emb = model.get_vision_embeddings(imgs=sat, isQ=False)

        gnd_embeds.append(gnd_emb.cpu())
        sat_embeds.append(sat_emb.cpu())

# ---------------------------------------------------------------
# Save
# ---------------------------------------------------------------
gnd_embeds = torch.cat(gnd_embeds, dim=0)
sat_embeds = torch.cat(sat_embeds, dim=0)

print("Ground shape    : " + str(gnd_embeds.shape))
print("Satellite shape : " + str(sat_embeds.shape))

torch.save(gnd_embeds, SAVE_DIR + '/clip_cvusa_gnd.pt')
torch.save(sat_embeds, SAVE_DIR + '/clip_cvusa_sat.pt')

print("DONE. Saved to " + SAVE_DIR)
print("Now set use_vis_embed=True and uncomment gnd/sat_embed_pretrn in attributes.py")
