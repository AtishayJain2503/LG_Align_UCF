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

        # Bind eval helper methods if missing (old Fahim models don't have them)
        if not hasattr(model, 'project_query'):
            import types
            def _project_query(self, xq):
                xq = self.vis_L1(xq);  xq = torch.relu(xq)
                xq = self.vis_L2(xq);  xq = torch.relu(xq)
                xq = self.vis_L3(xq)
                return xq
            model.project_query = types.MethodType(_project_query, model)

        # Override fuse_satellite if model lacks qformer (old arch used concat+MLP)
        if not hasattr(model, 'qformer'):
            import types
            print("  [INFO] No qformer found — binding concat+MLP fuse_satellite")
            def _fuse_satellite(self, xr, xt=None):
                if xt is not None:
                    xlt = torch.cat((xr, xt), dim=1)
                    xlt = self.mlp_txt(xlt)
                else:
                    xlt = xr
                xlt = self.vis_txt_L1(xlt);  xlt = torch.relu(xlt)
                xlt = self.vis_txt_L2(xlt);  xlt = torch.relu(xlt)
                xlt = self.vis_txt_L3(xlt)
                return xlt
            model.fuse_satellite = types.MethodType(_fuse_satellite, model)

        if not hasattr(model, 'encode_candidates'):
            import types
            def _encode_candidates(self, q, r, t):
                xq = self.get_vision_embeddings(imgs=q, isQ=True)
                xr = self.get_vision_embeddings(imgs=r, isQ=False)
                xt = self.get_text_embeddings(txt=t)
                return xq, xr, xt
            model.encode_candidates = types.MethodType(_encode_candidates, model)
    
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
