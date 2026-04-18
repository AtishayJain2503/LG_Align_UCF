import torch
from tqdm import tqdm
import time
import copy
import numpy as np
from torch.cuda.amp import autocast
import torch.nn.functional as F
from attributes import Configuration as hypm
import os

from helper_func import save_tensor, idsToDist
from torch.amp import autocast

def predict_embeddings(model, dataloader, dev=torch.device('cpu')):
    model.eval()
    
    xqs, xrs, xts, ids = [], [], [], []
    with torch.no_grad():
        for anchor, positive, negative, txt, idx in tqdm(dataloader, total=len(dataloader), desc="ViT Feature Extraction"):
            ids.append(idx)
            anchor, positive = anchor.to(dev), positive.to(dev)
            
            with autocast(device_type='cuda', enabled=hypm.use_mixed_precision):
                xq, xr, xt_pack = model.encode_candidates(anchor, positive, txt)
                
                if hypm.use_neg_text:
                    xt = xt_pack[0]
                else:
                    xt = xt_pack
                    
                xqs.append(xq)
                xrs.append(xr)
                xts.append(xt)
                
    xqs = torch.cat(xqs, dim=0)
    xrs = torch.cat(xrs, dim=0)
    xts = torch.cat(xts, dim=0)
    ids = torch.cat(ids, dim=0).to(dev)
    
    return xqs, xrs, xts, ids

def evaluate_fused(model, xqs, xrs, xts, topk=[1, 5, 10], batch_size=800):
    ts = time.time()
    N = xqs.shape[0]
    M = xrs.shape[0]
    
    topk_ext = list(topk)
    topk_ext.append(M//100)
    results = np.zeros([len(topk_ext)])
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(N), desc="O(N²) Fusion Eval"):
            xq_i = xqs[i:i+1] # [1, D]
            xt_i = xts[i:i+1] # [1, D_txt]
            
            sims_for_i = []
            
            # Chunk through references for this single query
            for start in range(0, M, batch_size):
                end = min(start + batch_size, M)
                xr_chunk = xrs[start:end]
                batch_len = xr_chunk.shape[0]
                
                xq_exp = xq_i.expand(batch_len, -1)
                xt_exp = xt_i.expand(batch_len, -1)
                
                with autocast(device_type='cuda', enabled=hypm.use_mixed_precision):
                    xq_proj, xlt, _ = model.fuse_and_project(xq_exp, xr_chunk, xt_exp)
                
                xq_norm = F.normalize(xq_proj, p=2, dim=1)
                xlt_norm = F.normalize(xlt, p=2, dim=1)
                
                sim = (xq_norm * xlt_norm).sum(dim=1)
                sims_for_i.append(sim)
                
            sims_for_i = torch.cat(sims_for_i, dim=0) # [M]
            
            gt_sim = sims_for_i[i]
            ranking = (sims_for_i > gt_sim).sum().item()
            
            for j, k in enumerate(topk_ext):
                if ranking < k:
                    results[j] += 1.
                    
    results = results / N * 100.
    print('Percentage-top1:{}, top5:{}, top10:{}, top1%:{}, time:{}'.format(results[0], results[1], results[2], results[-1], time.time() - ts))
    return results