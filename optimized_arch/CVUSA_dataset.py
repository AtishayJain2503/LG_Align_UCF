import os
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import json
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
import random
from transformers import CLIPProcessor, AutoProcessor, AutoTokenizer
from attributes import Configuration as hypm
import torch.nn.functional as F


def fov_crop_pano(img_pil, fov_deg=90, zero_pad=True):
    """Extract a random-azimuth FoV crop from an equirectangular panorama.

    Args:
        img_pil: PIL Image, full equirectangular panorama (W = 2*H expected)
        fov_deg: horizontal field of view in degrees (70–360)
        zero_pad: if True, embed crop in a zero canvas at its azimuth position
                  so CLIP positional encodings see correct spatial context.
    Returns:
        PIL Image ready for the CLIP processor.
    """
    img = TF.to_tensor(img_pil)  # [3, H, W]
    _, H, W = img.shape

    crop_w = max(1, int(W * fov_deg / 360.0))
    # Random starting azimuth
    start = random.randint(0, W - 1)

    # Seamless roll-wrap: move panorama so crop starts at 0, then slice
    rolled = torch.roll(img, -start, dims=2)  # [3, H, W]
    crop = rolled[:, :, :crop_w]              # [3, H, crop_w]

    if zero_pad:
        # Place crop back into a zero canvas at its original azimuth position
        canvas = torch.zeros_like(img)        # [3, H, W]
        end = min(start + crop_w, W)
        canvas[:, :, start:end] = crop[:, :, :(end - start)]
        # If crop wraps around the panorama boundary, handle the tail
        if start + crop_w > W:
            tail = (start + crop_w) - W
            canvas[:, :, :tail] = crop[:, :, (end - start):]
        out = TF.to_pil_image(canvas)
    else:
        out = TF.to_pil_image(crop)

    return out



class CVUSA_dataset_cropped(Dataset):
    def __init__(self, df, path, train=True, transform=None, lang='T1', TV=False):
        self.data_csv = df
        self.is_train = train
        self.transform = transform
        self.path = path
        self.lang = lang
        self.tokenizer = AutoTokenizer.from_pretrained(hypm.t_pretrain_weight)
        self.processor = AutoProcessor.from_pretrained(hypm.v_pretrain_weight)
        self.hard_neg_indices = None

        # if self.is_train:
        self.sat_images = self.data_csv.iloc[:, 0].values
        self.str_images = self.data_csv.iloc[:, 1].values
        self.index = self.data_csv.index.values
        self.data_csv["idx"] = self.data_csv[0].map(lambda x : int(x.split("/")[-1].split(".")[0]))

        if (self.is_train):
            self.T_lang = pd.read_csv(f'{self.path}/lang/gpt-4o/{self.lang}_train-19zl.csv')
            if(hypm.use_neg_text):
                self.T_lang_neg = pd.read_csv(f'{self.path}/lang/gpt-4o/{lang}_train-19zl_90_neg.csv')

        else:
            self.T_lang = pd.read_csv(f'{self.path}/lang/gpt-4o/{self.lang}_val-19zl.csv')
            if(hypm.use_neg_text):
                self.T_lang_neg = pd.read_csv(f'{self.path}/lang/gpt-4o/{self.lang}_val-19zl_90_neg.csv')

        
        if (TV):
            self.T_lang = pd.read_csv(f'{self.path}/lang/gpt-4o/T1_TV_all.csv')

    def update_hard_negatives(self, indices):
        self.hard_neg_indices = indices

    def __len__(self):
        return len(self.data_csv)
    def __getitem__(self, item):
        anchor_image_name = self.str_images[item]
        anchor_image_path = f"{self.path}/{anchor_image_name}"
        
        anchor_text = self.T_lang['Text'].loc[item]
        if(hypm.use_neg_text):
            anchor_text_neg = self.T_lang_neg['Text'].loc[item]
        ###### Anchor Image #######
        anchor_img = Image.open(anchor_image_path).convert('RGB')
        # if self.is_train:
        # anchor_label = self.labels[item]
        # positive_list = self.index[self.index!=item][self.str_images[self.index!=item]==anchor_image_name]
        # positive_item = random.choice(positive_list)
        positive_image_name = self.sat_images[item]
        positive_image_path = f"{self.path}/{positive_image_name}"
        positive_img = Image.open(positive_image_path).convert('RGB')
        #positive_img = self.images[positive_item].reshape(28, 28, 1)
        # negative_list = self.index[self.index!=item][self.sat_images[self.index!=item]!=positive_image_name]
        # negative_item = random.choice(negative_list)
        # negative_image_name = self.sat_images[negative_item]
        # negative_image_path = f"{self.path}/{negative_image_name}"
        # negative_img = Image.open(negative_image_path).convert('RGB')
        #negative_img = self.images[negative_item].reshape(28, 28, 1)

        if self.hard_neg_indices is not None:
            hn_idx = self.hard_neg_indices[item]
            hn_image_name = self.sat_images[hn_idx]
            hn_img = Image.open(f"{self.path}/{hn_image_name}").convert('RGB')
        else:
            # No hard negatives yet — pick a random different satellite
            rand_idx = (item + 1 + random.randint(0, len(self.sat_images) - 2)) % len(self.sat_images)
            hn_image_name = self.sat_images[rand_idx]
            hn_img = Image.open(f"{self.path}/{hn_image_name}").convert('RGB')

        if self.transform!=None and self.is_train:
            # Joint flip (preserves cardinal direction relationship)
            if random.random() > 0.5:
                anchor_img = TF.hflip(anchor_img)
                positive_img = TF.hflip(positive_img)
                hn_img = TF.hflip(hn_img)
            
            # Independent color jitter (lighting is view-specific)
            jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
            anchor_img = jitter(anchor_img)
            positive_img = jitter(positive_img)
            hn_img = jitter(hn_img)

            # CVUSA-C Robustness Stylizations (Histogram Equ, Autocontrast, Sharpness)
            # Crucial for 90-degree FoV where loss of context makes lighting/weather shifts fatal
            style_aug = transforms.Compose([
                transforms.RandomAutocontrast(p=0.3),
                transforms.RandomEqualize(p=0.3),
                transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.3)
            ])
            anchor_img = style_aug(anchor_img)
            positive_img = style_aug(positive_img)
            hn_img = style_aug(hn_img)

        # ------- Upgrade #2 + #3: FoV Crop with optional zero-padding -------
        if self.is_train and hypm.use_fov_aug:
            # random FoV between 70° and 360°
            fov1 = random.uniform(70, 360)
            fov2 = random.uniform(70, 360)  # second crop for ConGeo loss
            anchor_pil_1 = fov_crop_pano(anchor_img, fov_deg=fov1, zero_pad=hypm.use_zero_padding)
            anchor_pil_2 = fov_crop_pano(anchor_img, fov_deg=fov2, zero_pad=hypm.use_zero_padding)
        else:
            anchor_pil_1 = anchor_img
            anchor_pil_2 = anchor_img  # same crop at eval/no-aug time

        anchor_img_1 = self.processor(images=anchor_pil_1, return_tensors="pt").pixel_values.squeeze(0)
        anchor_img_2 = self.processor(images=anchor_pil_2, return_tensors="pt").pixel_values.squeeze(0)
        positive_img = self.processor(images=positive_img, return_tensors="pt").pixel_values.squeeze(0)
        negative_img = self.processor(images=hn_img, return_tensors="pt").pixel_values.squeeze(0)

        if(hypm.use_neg_text):
            return anchor_img_1, anchor_img_2, positive_img, negative_img, [anchor_text, anchor_text_neg], self.data_csv.idx[item]
        else:
            return anchor_img_1, anchor_img_2, positive_img, negative_img, anchor_text, self.data_csv.idx[item]

    

# class CVUSA_dataset_cropped(Dataset):
#     def __init__(self, df,path, train=True, transform=None):
#         self.data_csv = df
#         self.is_train = train
#         self.transform = transform
#         self.path = path
#         if self.is_train:
#             self.sat_images = df.iloc[:, 0].values
#             self.str_images = df.iloc[:, 1].values
#             self.index = df.index.values 
#     def __len__(self):
#         return len(self.data_csv)
#     def __getitem__(self, item):
#         anchor_image_name = self.str_images[item]
#         anchor_image_path = f"{self.path}/{anchor_image_name}"
#         ###### Anchor Image #######
#         anchor_img = Image.open(anchor_image_path).convert('RGB')
#         if self.is_train:
#             # anchor_label = self.labels[item]
#             # positive_list = self.index[self.index!=item][self.str_images[self.index!=item]==anchor_image_name]
#             # positive_item = random.choice(positive_list)
#             positive_image_name = self.sat_images[item]
#             positive_image_path = f"{self.path}/{positive_image_name}"
#             positive_img = Image.open(positive_image_path).convert('RGB')
#             #positive_img = self.images[positive_item].reshape(28, 28, 1)
#             negative_list = self.index[self.index!=item][self.sat_images[self.index!=item]!=positive_image_name]
#             negative_item = random.choice(negative_list)
#             negative_image_name = self.sat_images[negative_item]
#             negative_image_path = f"{self.path}/{negative_image_name}"
#             negative_img = Image.open(negative_image_path).convert('RGB')
#             #negative_img = self.images[negative_item].reshape(28, 28, 1)
#             if self.transform!=None:
#                 anchor_img = self.transform(anchor_img)
#                 positive_img = self.transform(positive_img)                   
#                 negative_img = self.transform(negative_img)
#         return anchor_img, positive_img, negative_img
    
class CVUSA_Dataset_Eval(Dataset):
    def __init__(self, df, path, train=True, transform=None, lang='T1', q_item=-1 ):
        self.data_csv = df
        self.is_train = train
        self.transform = transform
        self.path = path
        self.lang = lang
        self.tokenizer = AutoTokenizer.from_pretrained(hypm.t_pretrain_weight)
        self.processor = AutoProcessor.from_pretrained(hypm.v_pretrain_weight)
        self.q_item = q_item

        # if self.is_train:
        self.sat_images = df.iloc[:, 0].values
        self.str_images = df.iloc[:, 1].values
        self.index = df.index.values
        self.data_csv["idx"] = self.data_csv[0].map(lambda x : int(x.split("/")[-1].split(".")[0]))

        if (self.is_train):
            self.T_lang = pd.read_csv(f'{self.path}/lang/gpt-4o/{lang}_train-19zl.csv')

            if(hypm.use_neg_text):
                self.T_lang_neg = pd.read_csv(f'{self.path}/lang/gpt-4o/{lang}_train-19zl_90_neg.csv')

        else:
            self.T_lang = pd.read_csv(f'{self.path}/lang/gpt-4o/{lang}_val-19zl.csv')
            if(hypm.use_neg_text):
                self.T_lang_neg = pd.read_csv(f'{self.path}/lang/gpt-4o/{lang}_val-19zl_90_neg.csv')

        



    def __len__(self):
        return len(self.data_csv)
    def __getitem__(self, item):
        if(self.q_item >-1):
            anchor_image_name = self.str_images[self.q_item]
            anchor_image_path = f"{self.path}/{anchor_image_name}"
            
            anchor_text = self.T_lang['Text'].loc[self.q_item]
        else:
            raise Exception(f"{self.q_item} is wrong query item")

        if(hypm.use_neg_text):
            anchor_text_neg = self.T_lang_neg['Text'].loc[item]
        ###### Anchor Image #######
        anchor_img = Image.open(anchor_image_path).convert('RGB')
        positive_image_name = self.sat_images[item]
        positive_image_path = f"{self.path}/{positive_image_name}"
        positive_img = Image.open(positive_image_path).convert('RGB')
        
        if self.transform!=None:
            anchor_img = self.processor(images=anchor_img, return_tensors="pt")
            positive_img = self.processor(images=positive_img, return_tensors="pt")
            
            anchor_img = anchor_img.pixel_values
            anchor_img = torch.squeeze(anchor_img)

            positive_img = positive_img.pixel_values
            positive_img = torch.squeeze(positive_img)

            negative_img = positive_img



        if(hypm.use_neg_text):
            return anchor_img, positive_img, negative_img, [anchor_text, anchor_text_neg], self.data_csv.idx[item]
        else:
            return anchor_img, positive_img, negative_img, anchor_text, self.data_csv.idx[item]

# class CVUSA_Dataset_Eval(Dataset):
    
#     def __init__(self,
#                  data_folder,
#                  split,
#                  img_type,
#                  transforms=None,
#                  lang='T1'
#                  ):
        
#         super().__init__()
 
#         self.data_folder = data_folder
#         self.split = split
#         self.img_type = img_type
#         self.transforms = transforms
#         self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        
#         if split == 'train':
#             self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None)
#             if lang=='T1':
#                 self.df_lang = pd.read_csv(f'{data_folder}/lang/T1_train-19zl.csv')
#         else:
#             self.df = pd.read_csv(f'{data_folder}/splits/val-19zl.csv', header=None)
#             if lang=='T1':
#                 self.df_lang = pd.read_csv(f'{data_folder}/lang/T1_val-19zl.csv')

        
#         self.df = self.df.rename(columns={0:"sat", 1:"ground", 2:"ground_anno"})
        
#         self.df["idx"] = self.df.sat.map(lambda x : int(x.split("/")[-1].split(".")[0]))

#         self.idx2sat = dict(zip(self.df.idx, self.df.sat))
#         self.idx2ground = dict(zip(self.df.idx, self.df.ground))
   
    
#         if self.img_type == "reference":
#             self.images = self.df.sat.values
#             self.label = self.df.idx.values
            
#         elif self.img_type == "query":
#             self.images = self.df.ground.values
#             self.label = self.df.idx.values 
#         else:
#             raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")
                

#     def __getitem__(self, index):
        
#         # img = cv2.imread(f'{self.data_folder}/{self.images[index]}')
#         # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = Image.open(f'{self.data_folder}/{self.images[index]}').convert('RGB')
#         text = self.df_lang['Text'].loc[index]
        
#         # image transforms
#         if self.transforms is not None:
#             # img = self.transforms(img)
            
#             img = self.processor(images=img, return_tensors="pt")
#             img = img.pixel_values
#             img = torch.squeeze(img)

            
#         label = torch.tensor(self.label[index], dtype=torch.long)

#         return img, label, text


#         # if self.img_type == "query":    
#         #     return img, label, text
#         # else:
#         #     return img, label

#     def __len__(self):
#         return len(self.images)

            