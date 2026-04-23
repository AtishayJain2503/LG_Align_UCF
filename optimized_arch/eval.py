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

    xqs, xrs_pooled, xrs_seq, xts, ids = [], [], [], [], []
    with torch.no_grad():
        for anchor, anchor2, positive, negative, txt, idx in tqdm(dataloader, total=len(dataloader), desc="ViT Feature Extraction"):
            ids.append(idx)
            anchor, positive = anchor.to(dev), positive.to(dev)

            with autocast(device_type='cuda', enabled=hypm.use_mixed_precision):
                xq, xr, xt_pack = model.encode_candidates(anchor, positive, txt)

                if hypm.use_neg_text:
                    xt = xt_pack[0]
                else:
                    xt = xt_pack

            xqs.append(xq.cpu())
            if hypm.fusion_mode == 'qformer_patch':
                xrs_pooled.append(xr[0].cpu())
                xrs_seq.append(xr[1].cpu())
            else:
                xrs_pooled.append(xr.cpu())
            xts.append(xt.cpu())

    xqs = torch.cat(xqs, dim=0)   # [N, D_vis]
    xrs_pooled = torch.cat(xrs_pooled, dim=0)   # [N, D_vis]
    if len(xrs_seq) > 0:
        xrs_seq = torch.cat(xrs_seq, dim=0)
    xts = torch.cat(xts, dim=0)   # [N, D_txt]
    ids = torch.cat(ids, dim=0)

    if hypm.fusion_mode == 'qformer_patch':
        return xqs, (xrs_pooled, xrs_seq), xts, ids
    return xqs, xrs_pooled, xts, ids


def evaluate_fused(model, xqs, xrs, xts, topk=[1, 5, 10], batch_size=512):
    """
    Correct O(N^2) evaluation — matches Fahim's per-query eval logic:
      1. Project every GROUND image through query MLP head  → xq_proj [N, D]
      2. For EACH query[i]:
         - Fuse ALL satellites with query[i]'s text → xlt_i [M, D]
         - Rank which satellite is closest to query[i]

    Key insight: at inference time, you only have the QUERY's text description.
    You don't know which satellite corresponds to which location. So each
    satellite must be fused with the query's text, NOT its own.
    """
    ts   = time.time()
    N    = xqs.shape[0]
    if hypm.fusion_mode == 'qformer_patch':
        M = xrs[0].shape[0]
    else:
        M = xrs.shape[0]
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

        # ── Step 2: For EACH query, fuse ALL satellites with that query's text ─
        print("Computing per-query satellite fusion + ranking...")
        for qi in tqdm(range(N), desc="Ranking queries"):
            # Broadcast query[qi]'s text to all satellites
            xt_qi = xts[qi]  # [D_txt] — this single query's text embedding

            # Fuse every satellite with this query's text
            xlt_chunks = []
            for s in range(0, M, batch_size):
                e = min(s + batch_size, M)
                bs = e - s
                # Repeat this query's text for the whole satellite batch
                xt_batch = xt_qi.unsqueeze(0).expand(bs, -1).to(dev)

                with autocast(device_type='cuda', enabled=hypm.use_mixed_precision):
                    if hypm.fusion_mode == 'qformer_patch':
                        xr_batch = (xrs[0][s:e].to(dev), xrs[1][s:e].to(dev))
                    else:
                        xr_batch = xrs[s:e].to(dev)

                    fused = model.fuse_satellite(xr_batch, xt_batch)
                xlt_chunks.append(F.normalize(fused.float(), p=2, dim=1).cpu())
            xlt_qi = torch.cat(xlt_chunks, dim=0)  # [M, D]

            # Rank: similarity of query[qi] against all fused satellites
            sims = xq_proj[qi] @ xlt_qi.T  # [M]
            gt_sim = sims[qi].item()
            ranking = (sims > gt_sim).sum().item()

            for j, k in enumerate(topk_ext):
                if ranking < k:
                    results[j] += 1.

    results = results / N * 100.
    print('Percentage-top1:{}, top5:{}, top10:{}, top1%:{}, time:{}'.format(
        results[0], results[1], results[2], results[-1], time.time() - ts))
    return results