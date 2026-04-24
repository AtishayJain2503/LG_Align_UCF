import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import time
from datetime import datetime
from torch.utils.data import DataLoader
from CVACT_dataset import CVACT_dataset_cropped
from GAMa_dataset import GAMa_dataset_cropped
from VIGOR_dataset import VIGOR_dataset_cropped
from eval import predict_embeddings, evaluate_fused
from torchvision import transforms
from CVUSA_dataset import CVUSA_Dataset_Eval, CVUSA_dataset_cropped
from attributes import Configuration as hypm
from helper_func import write_to_file, write_to_rank_file, create_neg_keys, create_neg_keys_2, create_neg_keys_3
from torch.profiler import profile, ProfilerActivity

import torch.nn.functional as F
from torch.amp import autocast, GradScaler

def mine_hard_negatives(model, loader, dev, top_k=10):
    """Returns hard_neg_indices: LongTensor of shape [N]"""
    model.eval()
    q_feats, r_feats = [], []
    with torch.no_grad():
        for anchor, anchor2, positive, negative, txt, idx in tqdm(loader, desc="Mining"):
            anchor = anchor.to(dev)
            positive = positive.to(dev)
            with autocast(device_type='cuda', enabled=hypm.use_mixed_precision):
                qf = model.get_vision_embeddings(imgs=anchor, isQ=True)
                rf = model.get_vision_embeddings(imgs=positive, isQ=False)
            q_feats.append(F.normalize(qf, dim=-1).cpu())
            r_feats.append(F.normalize(rf, dim=-1).cpu())
    
    q_feats = torch.cat(q_feats)  # [N, D]
    r_feats = torch.cat(r_feats)  # [N, D]
    N = q_feats.shape[0]
    
    hard_neg_indices = torch.zeros(N, dtype=torch.long)
    chunk = 1000
    for i in range(0, N, chunk):
        q_chunk = q_feats[i:i+chunk]          # [chunk, D]
        sims = q_chunk @ r_feats.T            # [chunk, N]
        # mask true positives (diagonal of full matrix, offset by i)
        for j in range(q_chunk.shape[0]):
            sims[j, i + j] = -float('inf')
        # top-K hardest, pick one randomly
        _, top_k_idx = torch.topk(sims, k=top_k, dim=1)  # [chunk, K]
        rand = torch.randint(0, top_k, (q_chunk.shape[0],))
        hard_neg_indices[i:i+chunk] = top_k_idx[torch.arange(q_chunk.shape[0]), rand]
    
    model.train()
    return hard_neg_indices

def time_stamp():
    now = datetime.now()
    print(f'\nDate: {now}\n')



def train_step_eval(step=-1, mdl=None, dev='cpu' ):
    mdl.eval()
    print(f'\nTrain Step Eval: {step+1}\n')


    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225]),
    ])

    if(hypm.eval_db=="CVUSA"):
#--------------------------------CVUSA------------------------------------------
        data_path = hypm.data_path #don't include the / at the end

        val_data= pd.read_csv(f'{data_path}/splits/val-19zl.csv', header=None)
        # val_data= pd.read_csv(f'{data_path}/splits/val-19zl_panos.csv', header=None)

        val_ds = CVUSA_dataset_cropped(df = val_data, path=data_path, transform=transform, train=False, lang=hypm.lang)
        val_loader = DataLoader(val_ds, batch_size=hypm.batch_size, shuffle=False)
#--------------------------------CVACT------------------------------------------
    elif(hypm.eval_db=="CVACT"):
        data_path = hypm.data_path

        val_data= pd.read_csv(f'{data_path}/splits/CVACT_sm_val.csv')
        val_ds = CVACT_dataset_cropped(df = val_data, path=data_path, transform=transform, train=False, lang=hypm.lang)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
#--------------------------------VIGOR------------------------------------------
    elif(hypm.eval_db=="VIGOR"):
        data_path = hypm.data_path

        val_data= pd.read_csv(f'{data_path}/splits/VIGOR_test.csv')
        val_ds = VIGOR_dataset_cropped(df = val_data, path=data_path, transform=transform, train=False, lang=hypm.lang)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
#--------------------------------GAMa------------------------------------------
    elif(hypm.eval_db=="GAMa"):
        data_path = hypm.data_path

        val_data= pd.read_csv(f'{data_path}/split/gama_test.csv')
        val_ds = GAMa_dataset_cropped(df = val_data, path=data_path, transform=transform, train=False, lang=hypm.lang)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)


        print(f'\nNumber of Validation data: {val_data.shape[0]}')



    print("\nExtract Features:")
    xqs, xrs, xts, ids = predict_embeddings(model=mdl, dataloader=val_loader, dev=dev)

    print("Compute Scores:")
    r1 = evaluate_fused(model=mdl, xqs=xqs, xrs=xrs, xts=xts, topk=[1, 5, 10])
    
    write_to_file(expID=hypm.expID, msg=f'Train_eval_epoch: {step+1} => ', content=r1)
    write_to_rank_file(expID=hypm.expID, step=step, row=r1)

    print(r1)
    # mdl.train()



def train(model, criterion, optimizer, scheduler, train_loader, train_mining_loader, num_epochs=10, dev='cpu'):
    model.train()
    scaler = GradScaler()

    # wait before starting progress bar
    time.sleep(0.1)

    all_loses = []
    
    for epoch in range(num_epochs):
        model.train()

        # --- Phase 2: Unfreeze backbone at epoch 10 for joint fine-tuning ---
        if epoch == 10 and hypm.unfreeze_backbone:
            print("\n--- Phase 2: Unfreezing CLIP backbone for joint fine-tuning ---\n")
            for param in model.query.parameters():
                param.requires_grad = True
            for param in model.ref.parameters():
                param.requires_grad = True
            for param in model.text.parameters():
                param.requires_grad = True
            # Activate learning rate for the tracked backbone parameter group
            optimizer.param_groups[1]['lr'] = 1e-5
            # CRITICAL: Tell the scheduler this is the new base_lr for group 1.
            # Without this, CosineAnnealingLR still thinks base_lr=0.0 and will
            # overwrite lr back to ~0 on the very next scheduler.step() call.
            scheduler.base_lrs[1] = 1e-5
        
        if epoch >= 1 and (epoch % 3 == 0):
            hard_neg_indices = mine_hard_negatives(model, train_mining_loader, dev, top_k=10)
            train_loader.dataset.update_hard_negatives(hard_neg_indices)
            model.train()

        running_loss = []
        bar = tqdm(train_loader, total=len(train_loader))
        for anchor, anchor2, positive, hn_img, txt, idx in bar:
            
            optimizer.zero_grad()
            
            anchor  = anchor.to(dev)
            anchor2 = anchor2.to(dev)
            positive = positive.to(dev)
            hn_img = hn_img.to(dev)

            # anchor, positive, negative = anchor.to(torch.float16), positive.to(torch.float16), negative.to(torch.float16)

            with autocast(device_type='cuda', enabled=hypm.use_mixed_precision):
                anchor_embedding, positive_embedding, _ = model(q=anchor, r=positive, t=txt, isTrain=True, isQuery=True)
                
                # --- Upgrade #1: ConGeo self-supervised ground-ground loss ---
                if hypm.use_congeo_loss:
                    # anchor is already completely encoded by model() above
                    # and corresponds EXACTLY to the visual-only projection 
                    # in qformer fusion_mode. No need to run ViT twice!
                    anchor2_proj = model.project_query(
                        model.get_vision_embeddings(anchor2, isQ=True)
                    )
                    a1 = F.normalize(anchor_embedding, p=2, dim=1)
                    a2 = F.normalize(anchor2_proj, p=2, dim=1)
                    logits_cg = a1 @ a2.T / 0.07
                    labels_cg = torch.arange(a1.shape[0], device=dev)
                    congeo_loss = (
                        F.cross_entropy(logits_cg, labels_cg) +
                        F.cross_entropy(logits_cg.T, labels_cg)
                    ) * 0.5
                else:
                    congeo_loss = 0.0
                
                with torch.no_grad():
                    return_seq = (hypm.fusion_mode == 'qformer_patch')
                    if return_seq:
                        xr_pooled, xr_seq = model.get_vision_embeddings(hn_img, isQ=False, return_seq=True)
                        hn_raw = (xr_pooled, xr_seq)
                    else:
                        hn_raw = model.get_vision_embeddings(hn_img, isQ=False)
                        
                    if hypm.fusion_mode == 'none':
                        hn_emb = model.vis_txt_L3(torch.relu(model.vis_txt_L2(torch.relu(model.vis_txt_L1(hn_raw)))))
                    else:
                        # Vision-only abstraction for hard negative since we don't have its text
                        hn_emb = model.fuse_satellite(hn_raw, xt=None)

                if(hypm.use_neg_text):
                    neg_forward = hn_emb.unsqueeze(1) 
                    neg_reverse = torch.empty((hn_emb.shape[0], 0, hn_emb.shape[-1]), device=dev) # No reverse distractors
                else:
                    if train_loader.dataset.hard_neg_indices is not None:
                        neg_forward, neg_reverse = create_neg_keys_3(A=anchor_embedding, P=positive_embedding, NN=hn_emb)
                    else:
                        neg_forward, neg_reverse = create_neg_keys_2(A=anchor_embedding, P=positive_embedding)

                loss = criterion(query=anchor_embedding, positive_key=positive_embedding, negative_keys_forward=neg_forward, negative_keys_reverse=neg_reverse)
                loss = loss + hypm.congeo_weight * congeo_loss



            # anchor_embedding, positive_embedding = model(q = anchor, r = positive, isTrain = True, isQuery = True)
            # _, negative_embedding_combine = model(q = anchor, r = negative, isTrain = True, isQuery = True)
            # loss  = criterion(anchor_embedding, positive_embedding, negative_embedding_combine)
            # -----------------------------------------------------------------------------------------------------------
            # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            #     model(q = anchor, r = positive, t=txt, isTrain = False, isQuery = True)

            # # Sum FLOPs from the profiler and convert to GFLOPS
            # total_flops = sum([e.cpu_memory_usage for e in prof.key_averages()])
            # gflops = total_flops / 1e9

            # print(f"Estimated GFLOPS: {gflops:.4f}")
            # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            #     model(q = anchor, r = positive, t=txt, isTrain = False, isQuery = True)


            # # Sum FLOPs and convert to GFLOPS
            # total_flops = sum([e.cpu_memory_usage for e in prof.key_averages()])
            # gflops = total_flops / 1e9

            # print(f"Estimated GFLOPS: {gflops:.4f}")
            # -----------------------------------------------------------------------------------------------------------

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # total_loss += loss.item()
            running_loss.append(loss.cpu().detach().numpy())
            # print(f"Epoch {epoch+1}, Loss: {loss.item()}")

        print(f"Epoch: {epoch+1}/{num_epochs} Loss: {np.mean(running_loss)}")
        all_loses.append(np.mean(running_loss))

        write_to_file(expID=hypm.expID, msg=f'Loss_on_epoch:{epoch+1}=>', content=np.mean(running_loss))
        write_to_file(expID=hypm.expID, msg=f'LR_epoch:{epoch+1}=>', content=f'{scheduler.get_last_lr()[0]:.8f}')
        
        scheduler.step()

        # df_loss[epoch, "Loss"] = loss.cpu().detach().numpy()
        
        # ---------------------Training Evaluation---------------------------
        # if((epoch+1)%hypm.train_eval_per_epoch==0):
        #     train_step_eval(step=epoch, mdl=model, dev=dev)
        #     model.train()
        # ------------------------------------------------------------------
        info_filepath = f'info/info_{hypm.expID}.txt'
        with open(info_filepath, 'a') as file:
            file.write(f'\n**************Epoch: {epoch+1}**************\n')


        # ------------------------------------------------------------------
    
        if hypm.save_weights:
            torch.save(model.state_dict(), f'model_weights/{hypm.expID}/model_tr.pth')
    
    time_stamp()
    
    return all_loses

# def train(model, criterion, optimizer, train_loader, num_epochs=10, dev='cpu'):
#     model.train()
#     epoch_loss = []


#     time_stamp()
#     for epoch in range(num_epochs):
#         # total_loss = 0.0
#         print(f'Epoch#{epoch}')
#         running_loss = []
#         for i, (anchor, positive, negative) in enumerate(tqdm(train_loader)):
#             optimizer.zero_grad()
#             anchor, positive, negative = anchor.to(dev), positive.to(dev), negative.to(dev)
#             anchor_embedding = model(anchor, isQuery = True)
#             positive_embedding = model(positive, isQuery = False)
#             negative_embedding = model(negative, isQuery = False)
#             loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
#             loss.backward()
#             optimizer.step()
#             # total_loss += loss.item()
#             running_loss.append(loss.cpu().detach().numpy())
#             # print(f"Epoch {epoch+1}, Loss: {loss.item()}")
#         print(f"Epoch: {epoch+1}/{num_epochs} Loss: {np.mean(running_loss)}")
#         epoch_loss.append(np.mean(running_loss))
#         # df_loss[epoch, "Loss"] = loss.cpu().detach().numpy()
    
#     time_stamp()
    
#     return epoch_loss




# only for HuggingFace CLIP

# def train_step_eval(step=-1, query_features=None, reference_features=None, topk=[1,5,10] ):
#     print(f'Train Step Eval: {step}')
#     print("Compute Scores:")
#     if(query_features is not None and reference_features is not None):
#         N = query_features.shape[0]
#         M = reference_features.shape[0]
#         topk.append(M//100)
#         results = np.zeros([len(topk)])
#         # for CVUSA, CVACT
#         query_features = query_features.cpu()
#         reference_features = reference_features.cpu()
        
#         query_features = query_features.detach().numpy()
#         reference_features = reference_features.detach().numpy()



#         if N < 80000:
#             query_features_norm = np.sqrt(np.sum((query_features**2), axis=1, keepdims=True))
#             reference_features_norm = np.sqrt(np.sum((reference_features ** 2), axis=1, keepdims=True))
#             similarity = np.matmul(query_features/query_features_norm, (reference_features/reference_features_norm).T)
            
#             # print(similarity)
#             # save_tensor(var_name='similarity', var=similarity)
#             for i in range(N):
#                 # ranking = np.sum((similarity[i,:]>similarity[i,query_labels[i]])*1.)
#                 ranking = np.sum((similarity[i,:]>similarity[i,i])*1.)


#                 for j, k in enumerate(topk):
#                     if ranking < k:
#                         results[j] += 1.

#         results = results/ query_features.shape[0] * 100.
#         print('Percentage-top1:{}, top5:{}, top10:{}, top1%:{}'.format(results[0], results[1], results[2], results[-1]))
#     else:
#         print('problem with embedding')

