import torch
from tqdm import tqdm
import time
import numpy as np
import torch.nn.functional as F
from attributes import Configuration as hypm
from torch.amp import autocast


def predict_embeddings(model, dataloader, dev=torch.device('cpu')):
    """
    Mirrors Fahim's predict() exactly.

    For every sample i in the dataloader:
        anchor   = ground image i
        positive = satellite image i  (the CORRECT match, pre-paired in the dataset)
        txt      = text description of ground image i  (from T1_val-19zl.csv, row i)

    model() returns:
        query_feature[i] = project(ground_i)        -- pure visual, no text
        ref_feature[i]   = fuse(sat_i, text_i)      -- satellite enriched with ITS OWN text

    The resulting ref_features gallery is STATIC -- each satellite_j is always fused
    with text_j (its own paired ground description), NOT with any query's text.

    Ranking:  similarity[i, j] = query_feature[i] . ref_feature[j]
              Correct match is always on the diagonal (i == j).
    """
    model.eval()
    time.sleep(0.1)

    query_features_list = []
    ref_feature_list    = []
    ids_list            = []

    print("\nExtract Features (Paired Forward Pass):")
    with torch.no_grad():
        bar = tqdm(dataloader, total=len(dataloader), desc="Feature Extraction")
        for anchor, anchor2, positive, negative, txt, idx in bar:
            ids_list.append(idx)
            anchor   = anchor.to(dev)
            positive = positive.to(dev)

            with autocast(device_type='cuda', enabled=hypm.use_mixed_precision):
                # One forward pass — returns (query_emb, ref_emb, _)
                # ref_emb = fuse(sat_i, text_i)  uses each sample's OWN paired text
                query_feature, ref_feature, _ = model(
                    q=anchor, r=positive, t=txt, isTrain=False, isQuery=True
                )

            query_features_list.append(query_feature.float().cpu())
            ref_feature_list.append(ref_feature.float().cpu())

    query_features = torch.cat(query_features_list, dim=0)   # [N, D]
    ref_features   = torch.cat(ref_feature_list,   dim=0)    # [N, D]
    ids            = torch.cat(ids_list,            dim=0)    # [N]

    return query_features, ref_features, ids


def evaluate_fused(query_features, ref_features, topk=[1, 5, 10]):
    """
    Mirrors Fahim's accuracy() exactly.

    Single O(N^2) matrix multiply — no per-query fusion loop.

    similarity[i, j] = L2_norm(query_features[i]) . L2_norm(ref_features[j])
                     = project(ground_i) . fuse(sat_j, text_j)

    Ground-truth is always the diagonal: similarity[i, i] = project(ground_i) . fuse(sat_i, text_i)
    """
    ts = time.time()
    N  = query_features.shape[0]
    M  = ref_features.shape[0]

    topk_ext = list(topk)
    topk_ext.append(M // 100)
    results = np.zeros([len(topk_ext)])

    # L2-normalise in fp32 for numerical stability (matches Fahim's norm)
    q_norm = F.normalize(query_features.float(), p=2, dim=1).numpy()   # [N, D]
    r_norm = F.normalize(ref_features.float(),   p=2, dim=1).numpy()   # [N, D]

    print("\nCompute Scores (O(N^2) matrix multiply):")
    similarity = np.matmul(q_norm, r_norm.T)   # [N, N]

    for i in range(N):
        # Ground-truth is always at index i (diagonal) — dataset is ordered, no label lookup needed
        gt_sim  = similarity[i, i]
        ranking = np.sum((similarity[i, :] > gt_sim) * 1.)

        for j, k in enumerate(topk_ext):
            if ranking < k:
                results[j] += 1.

    results = results / N * 100.
    print('Percentage-top1:{}, top5:{}, top10:{}, top1%:{}, time:{}'.format(
        results[0], results[1], results[2], results[-1], time.time() - ts))
    return results