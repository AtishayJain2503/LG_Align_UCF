import sys
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from attributes import Configuration as hypm
from custom_models import CLIP_model
from CVUSA_dataset import CVUSA_dataset_cropped
from eval import predict_embeddings, evaluate_fused

def recover_and_evaluate(exp_id):
    print(f"--- Recovering weights for Experiment ID: {exp_id} ---")
    data_path = hypm.data_path
    
    val_data= pd.read_csv(f'{data_path}/splits/val-19zl.csv', header=None).reset_index(drop=True)
    val_ds = CVUSA_dataset_cropped(df=val_data, path=data_path, transform=None, train=False, lang=hypm.lang)
    
    # num_workers=0: safe for login nodes (no OpenBLAS thread limit issues)
    # For fast eval, use Option 2: srun with --gres=gpu:1 instead
    val_loader = DataLoader(val_ds, batch_size=hypm.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    torch.cuda.set_device(hypm.cuda_set_device)
    model = CLIP_model(embed_dim=hypm.embed_dim)
    model.to(hypm.device)
    
    weight_path = f"model_weights/{exp_id}/model_tr.pth"
    if not os.path.exists(weight_path):
        print(f"Error: Could not find weights at {weight_path}")
        return
        
    print(f"Loading weights from {weight_path}...")
    model.load_state_dict(torch.load(weight_path, map_location=hypm.device))
    
    print("\n--- Starting Evaluation Extraction ---")
    xqs, xrs, xts, ids = predict_embeddings(model=model, dataloader=val_loader, dev=hypm.device)
    
    print("\n--- Running Target Ranking ---")
    # Using batch size 64 for fused eval array routing
    results = evaluate_fused(model, xqs=xqs, xrs=xrs, xts=xts, topk=[1, 5, 10], batch_size=64)
    print("\n--- METRICS RECOVERED SUCCESSFULLY ---")
    print(f"R@1: {results[0]}, R@5: {results[1]}, R@10: {results[2]}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python recover_eval.py <ExpID>")
    else:
        recover_and_evaluate(sys.argv[1])
