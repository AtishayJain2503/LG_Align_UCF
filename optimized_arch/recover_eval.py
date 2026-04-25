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
    val_loader = DataLoader(val_ds, batch_size=hypm.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    torch.cuda.set_device(hypm.cuda_set_device)
    model = CLIP_model(embed_dim=hypm.embed_dim)
    model.to(hypm.device)
    
    weight_path = f"model_weights/{exp_id}/model_tr.pth"
    if not os.path.exists(weight_path):
        print(f"Error: Could not find weights at {weight_path}")
        return
        
    print(f"Loading weights from {weight_path}...")
    checkpoint = torch.load(weight_path, map_location=hypm.device, weights_only=False)

    if isinstance(checkpoint, dict):
        # New format: state_dict — load with strict=False to handle architecture changes
        result = model.load_state_dict(checkpoint, strict=False)
        if result.missing_keys:
            print(f"  [WARN] Missing keys (unused modules, ok): {len(result.missing_keys)} keys")
            for k in result.missing_keys[:5]:
                print(f"    - {k}")
        if result.unexpected_keys:
            print(f"  [WARN] Unexpected keys (old arch, ok): {len(result.unexpected_keys)} keys")
    else:
        # Old format: torch.save(model, ...) — use the loaded model directly
        print("Detected full-model checkpoint (old format). Using loaded model directly.")
        model = checkpoint
        model.to(hypm.device)
    
    print("\n--- Starting Evaluation ---")
    query_features, ref_features, ids = predict_embeddings(model=model, dataloader=val_loader, dev=hypm.device)
    
    print("\n--- Computing Recall ---")
    results = evaluate_fused(query_features=query_features, ref_features=ref_features, topk=[1, 5, 10])
    print("\n--- METRICS RECOVERED SUCCESSFULLY ---")
    print(f"R@1: {results[0]:.2f}, R@5: {results[1]:.2f}, R@10: {results[2]:.2f}, R@1%: {results[3]:.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python recover_eval.py <ExpID>")
    else:
        recover_and_evaluate(sys.argv[1])
