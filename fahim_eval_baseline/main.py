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
from eval import predict, accuracy, calculate_scores
import torch.nn.functional as F
import copy
import math
from pytorch_metric_learning import losses as LS
from helper_func import create_folders, get_rand_id, hyparam_info, save_exp, write_to_file, write_to_rank_file
from transformers import CLIPProcessor
from attributes import Configuration as hypm





transform = transforms.Compose([
    transforms.Resize((224, 240)),
    # transforms.RandomCrop(224),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], 
    #                      std=[0.229, 0.224, 0.225]),
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


#--------------------------------CVUSA------------------------------------------
if(hypm.dataset_nm=="CVUSA"):
    # data_path = '/media/fahimul/2B721C03261BDC8D/Research/datasets/CVUSA' #don't include the / at the end
    # data_path = '/home/fa947945/datasets/CVUSA_Cropped/CVUSA' #don't include the / at the end
    data_path = hypm.data_path #don't include the / at the end

    train_data= pd.read_csv(f'{data_path}/splits/train-19zl.csv', header=None)
    # train_data= pd.read_csv(f'{data_path}/splits/train-19zl_5.csv', header=None)
    # train_data= pd.read_csv(f'{data_path}/splits/train-19zl_30.csv', header=None)
    # train_data= pd.read_csv(f'{data_path}/splits/train-19zl_panos.csv', header=None)


    val_data= pd.read_csv(f'{data_path}/splits/val-19zl.csv', header=None)
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
    data_path = '/home/fahimul/Documents/Research/Dataset/CVACT/ANU_data_small' #don't include the / at the end

    train_data= pd.read_csv(f'{data_path}/splits/CVACT_sm_train.csv')
    val_data= pd.read_csv(f'{data_path}/splits/CVACT_sm_val.csv')

    # train_data= pd.read_csv(f'{data_path}/splits/CVACT_sm_train_temp.csv')
    # val_data= pd.read_csv(f'{data_path}/splits/CVACT_sm_val_temp.csv')

    # train_data= pd.read_csv(f'{data_path}/splits/CVACT_sm_train_panos.csv')
    # val_data= pd.read_csv(f'{data_path}/splits/CVACT_sm_val_panos.csv')

    #--------------------------------CVACT------------------------------------------
    train_ds = CVACT_dataset_cropped(df = train_data, path=data_path, transform=transform, train=True, lang=hypm.lang)
    val_ds = CVACT_dataset_cropped(df = val_data, path=data_path, transform=transform, train=False, lang=hypm.lang)

#--------------------------------VIGOR------------------------------------------
elif(hypm.dataset_nm=="VIGOR"):
    data_path = '/home/fahimul/Documents/Research/Dataset/VIGOR' #don't include the / at the end

    train_data= pd.read_csv(f'{data_path}/splits/VIGOR_train.csv')
    # train_data= pd.read_csv(f'{data_path}/splits/VIGOR_train_temp.csv')

    val_data= pd.read_csv(f'{data_path}/splits/VIGOR_test.csv')





    #--------------------------------VIGOR------------------------------------------
    train_ds = VIGOR_dataset_cropped(df = train_data, path=data_path, transform=transform, train=True, lang=hypm.lang)
    val_ds = VIGOR_dataset_cropped(df = val_data, path=data_path, transform=transform, train=False, lang=hypm.lang)

#--------------------------------GAMa------------------------------------------
elif(hypm.dataset_nm=="GAMa"):
    data_path = '/home/fahimul/Documents/Research/Dataset/GAMa' #don't include the / at the end

    train_data= pd.read_csv(f'{data_path}/split/gama_train.csv')
    val_data= pd.read_csv(f'{data_path}/split/gama_test.csv')


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
    torch.cuda.set_device(hypm.cuda_set_device)





    create_folders()
    # print(f"Device: {device}")


    train_loader = DataLoader(train_ds, batch_size=hypm.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=hypm.batch_size, shuffle=False)
    # val_loader_ref = DataLoader(val_ref, batch_size=hypm.batch_size, shuffle=False)

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
    # model = torch.load(f'model_weights/{cvusa_pre_t_weight}/model_tr.pth').to(hypm.device) #for CVUSA
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


    optimizer = optim.Adam(parameters, lr=hypm.lr)
    # optimizer = optim.AdamW(parameters, lr=lr)
    # optimizer = optim.SGD(parameters, lr=lr)
    # ---------------------------------------------------------------


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
    all_loses = train(model, criterion, optimizer, train_loader, num_epochs=hypm.epochs, dev=hypm.device)
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

    # ***********************************************************************************************
    result = [0,0,0,0]
    val_qitemData= pd.read_csv(f'{data_path}/splits/val-19zl.csv', header=None)
    hypm.eval_size = val_qitemData.shape[0]  

    write_to_file(expID=hypm.expID, msg=f'number of q_item: {hypm.eval_size} \n', content='')
    # print(f"number of q_items: {val_qitemData.shape[0]}")
    for q_item in range(val_qitemData.shape[0]):
        print(f"q_item: {q_item}")
        hypm.batch_no=0
        val_qitem = CVUSA_Dataset_Eval(df = val_qitemData, path=data_path, transform=transform, train=False, lang=hypm.lang, q_item=q_item)
        val_qitem_loader = DataLoader(val_qitem, batch_size=hypm.batch_size, shuffle=False, num_workers=hypm.num_workers, pin_memory=True)
        query_features, reference_features, labels = predict(model=model, dataloader=val_qitem_loader, dev=hypm.device, isQuery=True)
        r1 =  accuracy(query_features=query_features, reference_features=reference_features, query_labels=labels, topk=[1, 5, 10], q_item=q_item)
        for i in range(len(result)):
            result[i] += r1[i]

        if(q_item%100==0 and q_item>0):
            temp_res = [(v / q_item) * 100 for v in result]
            write_to_file(expID=hypm.expID, msg=f'result on {q_item} => ', content=f"{temp_res}")

    # results = result/val_qitemData.shape[0] * 100.
    results = [(v / val_qitemData.shape[0]) * 100 for v in result]
    print(results)
    write_to_file(expID=hypm.expID, msg=f'final result on {q_item} => ', content=f"{results}")
    write_to_file(expID=hypm.expID, msg=f'End of training: ', content=datetime.now())

    # ***********************************************************************************************
    # print(f'{r1}\n') 


    write_to_file(expID=hypm.expID, msg=f'Final eval: ', content=r1)
    write_to_rank_file(expID=hypm.expID, step=hypm.epochs, row=r1)



    if hypm.save_weights:
        torch.save(model, f'model_weights/{hypm.expID}/model_tr.pth')
    





    torch.cuda.empty_cache()
        






if __name__ == '__main__':
    main()
