import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from main import transform
from attributes import Configuration as hypm
from CVUSA_dataset import CVUSA_Dataset_Eval
from tqdm import tqdm
import time
import os

def extract_features(model, dataloader, dev):
    model.eval()
    query_features_list = []
    ref_feature_list = []
    
    with torch.no_grad():
        for anchor, positive, negative, txt, idx in tqdm(dataloader, desc="Extracting features..."):
            anchor, positive, negative = anchor.to(dev), positive.to(dev), negative.to(dev)
            
            # Using Fahim's model exact forward pass to extract Ground features
            q_feat, _, _ = model(q=anchor, r=positive, t=txt, isTrain=False, isQuery=True)
            # Using Fahim's exact pass for Satellite features
            r_feat, _, _ = model(q=anchor, r=positive, t=txt, isTrain=False, isQuery=False)
            
            query_features_list.append(q_feat.cpu())
            ref_feature_list.append(r_feat.cpu())
            
    query_features = torch.cat(query_features_list, dim=0)
    ref_features = torch.cat(ref_feature_list, dim=0)
    
    return query_features, ref_features

def evaluate_fast(query_features, reference_features, topk=[1, 5, 10]):
    N = query_features.shape[0]
    results = np.zeros([len(topk)])
    
    # Normalize features
    q_norm = np.sqrt(np.sum((query_features.numpy()**2), axis=1, keepdims=True))
    r_norm = np.sqrt(np.sum((reference_features.numpy()**2), axis=1, keepdims=True))
    
    q_feat = query_features.numpy() / q_norm
    r_feat = reference_features.numpy() / r_norm
    
    # Compute full similarity matrix all at once
    print("Computing O(N^2) similarity matrix...")
    similarity = np.matmul(q_feat, r_feat.T)
    
    for i in tqdm(range(N), desc="Ranking"):
        # similarity[i, i] is the correct match in CVUSA
        ranking = np.sum((similarity[i, :] > similarity[i, i]) * 1.0)
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.0
                
    results = results / N * 100.
    return results

def main():
    print("CUDA device:", hypm.cuda_set_device)
    
    # Load model checkpoint
    model_path = f'model_weights/76663635/model_tr.pth'
    if not os.path.exists(model_path):
        print(f"ERROR: Could not find trained model at {model_path}")
        return
        
    print(f"Loading trained weights from {model_path}...")
    model = torch.load(model_path, map_location=hypm.device)
    model.eval()

    # Load dataset
    data_path = hypm.data_path
    val_df = pd.read_csv(f'{data_path}/splits/val-19zl.csv', header=None)
    
    # We load q_item=-1 to get ALL items sequentially without repeating 8884 times
    val_dataset = CVUSA_Dataset_Eval(df=val_df, path=data_path, transform=transform, train=False, lang=hypm.lang, q_item=-1)
    # Batch size 64 to speed up extraction
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=hypm.num_workers, pin_memory=True)
    
    print(f"Total Validation size: {len(val_dataset)}")
    
    # 1. Extract all features once O(N)
    q_feats, r_feats = extract_features(model, val_loader, hypm.device)
    
    # 2. Evaluate mathematically exact R@k O(N^2)
    metrics = evaluate_fast(q_feats, r_feats)
    
    print("\n--- FINAL METRICS FOR FAHIM BASELINE ---")
    print(f"Recall@1:  {metrics[0]:.2f}%")
    print(f"Recall@5:  {metrics[1]:.2f}%")
    print(f"Recall@10: {metrics[2]:.2f}%")

if __name__ == '__main__':
    main()
