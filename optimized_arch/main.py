from datetime import datetime
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" #new eval time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time
# from torch.cuda.amp import autocast
import numpy as np
from GAMa_dataset import GAMa_dataset_cropped
from VIGOR_dataset import VIGOR_dataset_cropped
from CVACT_dataset import CVACT_dataset_cropped
from CVUSA_dataset import CVUSA_dataset_cropped, CVUSA_Dataset_Eval
# from CVUSA_dataset import CVUSA_Dataset_Eval
from custom_models import ResNet, VIT, CLIP_model
from losses import Contrastive_loss, SoftTripletBiLoss, InfoNCE, InfoNCE_2
from train import train
from eval import predict, accuracy, calculate_scores, predict_embeddings, evaluate_fused
import torch.nn.functional as F
import copy
import math
from pytorch_metric_learning import losses as LS
from helper_func import create_folders, get_rand_id, hyparam_info, save_exp, write_to_file, write_to_rank_file
from transformers import CLIPProcessor
from attributes import Configuration as hypm





transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 240)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
])
# cvusa_pre_t_weight = 31985602 #vit-large-patch14+3 MLP
# cvact_pre_t_weight = 32583602 #vit-large-patch14+3 MLP
# vigor_pre_t_weight = 32583657 #vit-large-patch14+3 MLP

# cvusa_pre_t_weight = 40860473 #vit-large-patch14+3 MLP+adapter
# cvact_pre_t_weight = 40560627 #vit-large-patch14+3 MLP+adapter
# vigor_pre_t_weight = 40560717 #vit-large-patch14+3 MLP+adapter

cvusa_pre_t_weight = 54284102 #vit-large-patch14+3 MLP+adapter
cvact_pre_t_weight = 52442361 #vit-large-patch14+3 MLP+adapter
vigor_pre_t_weight = 52400590 #vit-large-patch14+3 MLP+adapter


import os

#--------------------------------CVUSA------------------------------------------
if(hypm.dataset_nm=="CVUSA"):
    data_path = hypm.data_path #don't include the / at the end

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"\n[CRITICAL ERROR]: The dataset path '{data_path}' was not found!\n"
                                f"Please update 'data_path' in attributes.py to point to your exact dataset folder on the cluster.\n"
                                f"Job halted to save cluster resources.")

    train_data= pd.read_csv(f'{data_path}/splits/train-19zl.csv', header=None).reset_index(drop=True)
    # train_data= pd.read_csv(f'{data_path}/splits/train-19zl_5.csv', header=None)
    # train_data= pd.read_csv(f'{data_path}/splits/train-19zl_30.csv', header=None)
    # train_data= pd.read_csv(f'{data_path}/splits/train-19zl_panos.csv', header=None)


    val_data= pd.read_csv(f'{data_path}/splits/val-19zl.csv', header=None).reset_index(drop=True)
    # val_data= pd.read_csv(f'{data_path}/splits/val-19zl_panos.csv', header=None)
    # val_data= pd.read_csv(f'{data_path}/splits/val-19zl_5.csv', header=None)



    df_loss = pd.DataFrame(columns=['Loss'])



    train_ds = CVUSA_dataset_cropped(df = train_data, path=data_path, transform=transform, train=True, lang=hypm.lang)
    val_ds = CVUSA_dataset_cropped(df = val_data, path=data_path, transform=transform, train=False, lang=hypm.lang)

    # val_que = CVUSA_Dataset_Eval(data_folder=data_path, split='val', img_type='query', transforms=transform)
    # val_ref = CVUSA_Dataset_Eval(data_folder=data_path, split='val', img_type='reference', transforms=transform)

    # hypm.latlong_csv = pd.read_csv(f"{data_path}/split_locations/all.csv")

    # tv_all = pd.read_csv(f"{data_path}/split_locations/tv_all.csv", header=None)
    # tv_all_ds = CVUSA_dataset_cropped(df = tv_all, path=data_path, transform=transform, train=False, lang=hypm.lang, TV=True)




#--------------------------------CVACT------------------------------------------
elif(hypm.dataset_nm=="CVACT"):
    data_path = hypm.data_path

    train_data= pd.read_csv(f'{data_path}/splits/CVACT_sm_train.csv').reset_index(drop=True)
    val_data= pd.read_csv(f'{data_path}/splits/CVACT_sm_val.csv').reset_index(drop=True)

    # train_data= pd.read_csv(f'{data_path}/splits/CVACT_sm_train_temp.csv')
    # val_data= pd.read_csv(f'{data_path}/splits/CVACT_sm_val_temp.csv')

    # train_data= pd.read_csv(f'{data_path}/splits/CVACT_sm_train_panos.csv')
    # val_data= pd.read_csv(f'{data_path}/splits/CVACT_sm_val_panos.csv')

    #--------------------------------CVACT------------------------------------------
    train_ds = CVACT_dataset_cropped(df = train_data, path=data_path, transform=transform, train=True, lang=hypm.lang)
    val_ds = CVACT_dataset_cropped(df = val_data, path=data_path, transform=transform, train=False, lang=hypm.lang)

#--------------------------------VIGOR------------------------------------------
elif(hypm.dataset_nm=="VIGOR"):
    data_path = hypm.data_path

    train_data= pd.read_csv(f'{data_path}/splits/VIGOR_train.csv').reset_index(drop=True)
    # train_data= pd.read_csv(f'{data_path}/splits/VIGOR_train_temp.csv')

    val_data= pd.read_csv(f'{data_path}/splits/VIGOR_test.csv').reset_index(drop=True)





    #--------------------------------VIGOR------------------------------------------
    train_ds = VIGOR_dataset_cropped(df = train_data, path=data_path, transform=transform, train=True, lang=hypm.lang)
    val_ds = VIGOR_dataset_cropped(df = val_data, path=data_path, transform=transform, train=False, lang=hypm.lang)

#--------------------------------GAMa------------------------------------------
elif(hypm.dataset_nm=="GAMa"):
    data_path = hypm.data_path

    train_data= pd.read_csv(f'{data_path}/split/gama_train.csv').reset_index(drop=True)
    val_data= pd.read_csv(f'{data_path}/split/gama_test.csv').reset_index(drop=True)


    #--------------------------------GAMa------------------------------------------
    train_ds = GAMa_dataset_cropped(df = train_data, path=data_path, transform=transform, train=True, lang=hypm.lang)
    val_ds = GAMa_dataset_cropped(df = val_data, path=data_path, transform=transform, train=False, lang=hypm.lang)

else:
    raise Exception('Dataset not found!!!')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embed_dim = 512
    lr = 0.000001
    batch_size = 32
    epochs = 100
    expID = get_rand_id()
    loss_margin = 1

    hypm.expID = expID
    hypm.msg = f'Run-{hypm.fusion_mode}; fusion={hypm.fusion_mode}; lang={hypm.lang}; dataset={hypm.dataset_nm}; lr={hypm.lr}; bs={hypm.batch_size}'
    torch.cuda.set_device(hypm.cuda_set_device)





    create_folders()
    train_loader = DataLoader(train_ds, batch_size=hypm.batch_size, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)
    train_mining_loader = DataLoader(train_ds, batch_size=hypm.batch_size, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=hypm.batch_size, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)
    # val_loader_ref = DataLoader(val_ref, batch_size=hypm.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # **********************************Only for CVUSA*****************************************
    # tv_all_loader = DataLoader(tv_all_ds, batch_size=hypm.batch_size, shuffle=False)
    # ******************************************************************************************

    if hypm.save_weights:
        os.mkdir(f'model_weights/{hypm.expID}')

    # model = ResNet(emb_dim=embed_dim).to(device)
    # model_r = ResNet(emb_dim=embed_dim).to(device)
    # model_q = ResNet(emb_dim=embed_dim).to(device)

    # model = ResNet().to(device)
    # model = VIT().to(device)
    # ---------------------------------------------------------------
    model = CLIP_model(embed_dim=hypm.embed_dim)
    # print(model)

    # -------------------------------EVAL--------------------------------
    # model = torch.load(f'model_weights/75845657/model_tr.pth', weights_only=False).to(hypm.device) #for CVUSA
    # model = torch.load(f'model_weights/{cvact_pre_t_weight}/model_tr.pth').to(hypm.device) #for CVACT
    # model = torch.load(f'model_weights/{vigor_pre_t_weight}/model_tr.pth').to(hypm.device) #for VIGOR

    # model = torch.load(f'model_weights/41286489/model_tr.pth').to(hypm.device) #for Resnet50



    # ---------------------------------------------------------------

    # torch.save(model, f'model_weights/{expID}/model_st.pth')
    

    # criterion = TripletLoss(margin=loss_margin)
    # criterion = nn.TripletMarginLoss(margin=0.5)
  
    # criterion = SoftTripletBiLoss()
    # ----------------------------LOSS_OG-----------------------------------

    # loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=hypm.label_smoothing)
    # criterion = InfoNCE(loss_function=loss_fn,
    #                         device=hypm.device,
    #                         )


    # parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    # ----------------------------------------------------------------------
    # ----------------------------LOSS_InfoNCE_2-------------------------------

    criterion = InfoNCE_2()


    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    
    # -----------------------Param info----------------------------------------
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)


    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Trainable Parameters: {trainable_params:,}")
    
    # ---------------------------------------------------------------
    # 2-Phase optimizer: Phase 1 trains only MLP heads at high LR.
    # Phase 2 (epoch 10+) adds the unfrozen backbone at low LR.
    # ---------------------------------------------------------------
    mlp_params = (
        list(model.vis_L1.parameters()) + list(model.vis_L2.parameters()) +
        list(model.vis_L3.parameters()) + list(model.vis_txt_L1.parameters()) +
        list(model.vis_txt_L2.parameters()) + list(model.vis_txt_L3.parameters())
    )
    optimizer = optim.AdamW(mlp_params, lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hypm.epochs, eta_min=1e-5
    )


    hypm.eval_size = val_data.shape[0]  
    
    print(f"CUDA device: {hypm.cuda_set_device}")  
    hyparam_info(emb_dim = hypm.embed_dim, 
                 loss_id = hypm.expID, 
                 ln_rate = hypm.lr, 
                 batch = hypm.batch_size, 
                 epc = hypm.epochs, 
                 ls_mrgn = hypm.loss_margin, 
                 trn_sz = train_data.shape[0],
                 val_sz= val_data.shape[0],
                 mdl_nm = model.modelName)
    
    save_exp(emb_dim=hypm.embed_dim, 
                loss_id=hypm.expID, 
                ln_rate=hypm.lr, 
                batch=hypm.batch_size, 
                epc=hypm.epochs, 
                ls_mrgn=hypm.loss_margin,
                lbl_sm=hypm.label_smoothing,
                dt_nm=hypm.dataset_nm, 
                trn_sz=train_data.shape[0],
                val_sz= val_data.shape[0],
                mdl_nm=hypm.v_pretrain_weight,
                msg= hypm.msg,
                adp_nm=hypm.v_adapter_id)
    
    # write_to_file(expID=hypm.expID, msg=f'Hyperparameter info: ', content=datetime.now())

    # for key, value in vars(hypm).items():
    #     if not key.startswith("__"):  # Exclude built-in attributes
    #         print(f"{key}: {value}")
    #         write_to_file(expID=hypm.expID, msg=f'{key}: ', content=f'{value}')

    # ***************************************Training************************************************
    write_to_file(expID=hypm.expID, msg="Trainable Parameters: ", content=f'{trainable_params:,}\n')

    print("Training Start")
    all_loses = train(model, criterion, optimizer, scheduler, train_loader, train_mining_loader, num_epochs=hypm.epochs, dev=hypm.device)
    df_loss = pd.DataFrame({'Loss': all_loses})
    df_loss.to_csv(f'losses/losses_{hypm.expID}.csv')

    write_to_file(expID=hypm.expID, msg=f'End of training: ', content=datetime.now())
    # ***********************************************************************************************


    print("\nExtract Features:")
    # query_features, reference_features, labels = predict(model=model, dataloader=val_loader, dev=hypm.device, isQuery=True)# og
    # reference_features, reference_labels = predict(model = model, dataloader=val_loader_ref, dev=hypm.device, isQuery=False)
    
    # print('TV_all Extract Features')
    # tv_query_features, tv_reference_features, labels = predict(model=model, dataloader=tv_all_loader, dev=hypm.device, isQuery=True)
     
    


    print("Compute Scores:")
    # r1 =  calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=1000, ranks=[1, 5, 10])
    # r1 =  accuracy(query_features=query_features, reference_features=reference_features, query_labels=labels, topk=[1, 5, 10])# og

    # latlong distance_calculation
    # r1 =  accuracy(query_features=query_features, reference_features=reference_features, query_labels=labels, topk=[1, 5, 10], tv_all_reference_features=tv_reference_features)

    if(hypm.save_vis_embed):
        all_gnd_embeddings = torch.cat(hypm.gnd_embed, dim=0)
        all_sat_embeddings = torch.cat(hypm.sat_embed, dim=0)

        print(f'gnd embed shape: {all_gnd_embeddings.shape}')
        print(f'sat embed shape: {all_sat_embeddings.shape}')


        torch.save(all_gnd_embeddings, "embeddings/clip_cvusa_gnd.pt")
        torch.save(all_sat_embeddings, "embeddings/clip_cvusa_sat.pt")

        print(f"Saved embeddings Ground and Satelllite")

    print("\nExtract Features (Decoupled ViT):")
    xqs, xrs, xts, ids = predict_embeddings(model=model, dataloader=val_loader, dev=hypm.device)
    
    print("Compute Scores (O(N^2) Evaluated in batch):")
    r1 = evaluate_fused(model=model, xqs=xqs, xrs=xrs, xts=xts, topk=[1, 5, 10])

    # accuracy() already converts to percentages internally
    results = r1

    write_to_file(expID=hypm.expID, msg=f'final result => ', content=f"{results}")
    write_to_rank_file(expID=hypm.expID, step=hypm.epochs, row=r1)



    if hypm.save_weights:
        torch.save(model, f'model_weights/{hypm.expID}/model_tr.pth')
    





    torch.cuda.empty_cache()
        






if __name__ == '__main__':
    main()
