import torch
from tqdm import tqdm
import time
import numpy as np
import torch.nn.functional as F
from attributes import Configuration as hypm
from helper_func import save_tensor, idsToDist
from torch.amp import autocast


def predict_embeddings(model, dataloader, dev=torch.device('cpu')):
    """
    Extract raw ViT embeddings (before MLP) for all samples.
    Returns:
      xqs  [N, D] - raw ground (query) ViT embeddings
      xrs  [N, D] - raw satellite (reference) ViT embeddings
      xts  [N, D] - text embeddings for each location (paired with its satellite)
      ids  [N]    - sample IDs
    """
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

            xqs.append(xq.cpu())
            xrs.append(xr.cpu())
            xts.append(xt.cpu())

    xqs = torch.cat(xqs, dim=0)   # [N, D_vis]
    xrs = torch.cat(xrs, dim=0)   # [N, D_vis]
    xts = torch.cat(xts, dim=0)   # [N, D_txt]
    ids = torch.cat(ids, dim=0)

    return xqs, xrs, xts, ids


def evaluate_fused(model, xqs, xrs, xts, topk=[1, 5, 10], batch_size=512):
    """
    Correct O(N^2) evaluation:
      1. Project every GROUND image through query MLP head  → xq_proj [N, D]
      2. Fuse every SATELLITE with its OWN text through sat MLP → xlt [N, D]
         (each satellite uses xts[j], NOT the query's text)
      3. Full cosine similarity matrix and rank each query.

    This fixes the text-assignment bug where the old code used the query's
    text for every satellite reference during evaluation.
    """
    ts   = time.time()
    N    = xqs.shape[0]
    M    = xrs.shape[0]
    dev  = next(model.parameters()).device

    topk_ext = list(topk)
    topk_ext.append(M // 100)
    results = np.zeros([len(topk_ext)])

    model.eval()
    with torch.no_grad():

        # ── Step 1: project ALL query (ground) embeddings ─────────────────────
        xq_proj_list = []
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            with autocast(device_type='cuda', enabled=hypm.use_mixed_precision):
                proj = model.project_query(xqs[s:e].to(dev))
            xq_proj_list.append(F.normalize(proj.float(), p=2, dim=1).cpu())
        xq_proj = torch.cat(xq_proj_list, dim=0)   # [N, D]

        # ── Step 2: fuse ALL satellite embeddings with their OWN text ─────────
        xlt_list = []
        for s in range(0, M, batch_size):
            e = min(s + batch_size, M)
            with autocast(device_type='cuda', enabled=hypm.use_mixed_precision):
                fused = model.fuse_satellite(xrs[s:e].to(dev), xts[s:e].to(dev))
            xlt_list.append(F.normalize(fused.float(), p=2, dim=1).cpu())
        xlt = torch.cat(xlt_list, dim=0)           # [M, D]

        # ── Step 3: chunked cosine similarity + ranking ────────────────────────
        print("Computing similarity matrix...")
        for s in tqdm(range(0, N, batch_size), desc="Ranking queries"):
            e      = min(s + batch_size, N)
            sims   = xq_proj[s:e] @ xlt.T          # [chunk, M]  — cosine (both normalised)
            for local_i in range(sims.shape[0]):
                global_i = s + local_i
                gt_sim   = sims[local_i, global_i].item()
                ranking  = (sims[local_i] > gt_sim).sum().item()
                for j, k in enumerate(topk_ext):
                    if ranking < k:
                        results[j] += 1.

    results = results / N * 100.
    print('Percentage-top1:{}, top5:{}, top10:{}, top1%:{}, time:{}'.format(
        results[0], results[1], results[2], results[-1], time.time() - ts))
    return results