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
from eval import predict, accuracy
from torchvision import transforms
from CVUSA_dataset import CVUSA_Dataset_Eval, CVUSA_dataset_cropped
from eval import accuracy, predict
from attributes import Configuration as hypm
from helper_func import write_to_file, write_to_rank_file, create_neg_keys, create_neg_keys_2, create_neg_keys_3
from torch.profiler import profile, ProfilerActivity


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
        data_path = '/home/fahimul/Documents/Research/Dataset/CVACT/ANU_data_small' #don't include the / at the end

        val_data= pd.read_csv(f'{data_path}/splits/CVACT_sm_val.csv')
        val_ds = CVACT_dataset_cropped(df = val_data, path=data_path, transform=transform, train=False, lang=hypm.lang)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
#--------------------------------VIGOR------------------------------------------
    elif(hypm.eval_db=="VIGOR"):
        data_path = '/home/fahimul/Documents/Research/Dataset/VIGOR' #don't include the / at the end

        val_data= pd.read_csv(f'{data_path}/splits/VIGOR_test.csv')
        val_ds = VIGOR_dataset_cropped(df = val_data, path=data_path, transform=transform, train=False, lang=hypm.lang)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
#--------------------------------GAMa------------------------------------------
    elif(hypm.eval_db=="GAMa"):
        data_path = '/home/fahimul/Documents/Research/Dataset/GAMa' #don't include the / at the end

        val_data= pd.read_csv(f'{data_path}/split/gama_test.csv')
        val_ds = GAMa_dataset_cropped(df = val_data, path=data_path, transform=transform, train=False, lang=hypm.lang)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)


        print(f'\nNumber of Validation data: {val_data.shape[0]}')



    print("\nExtract Features:")
    # query_features, query_labels = predict(model=mdl, dataloader=val_loader_que, dev=dev, isQuery=True)
    # reference_features, reference_labels = predict(model = mdl, dataloader=val_loader_ref, dev=dev, isQuery=False) 
    query_features, reference_features, labels = predict(model=mdl, dataloader=val_loader, dev=dev, isQuery=True)


    print("Compute Scores:")
    r1 =  accuracy(query_features=query_features, reference_features=reference_features, query_labels=labels, topk=[1, 5, 10])
    
    write_to_file(expID=hypm.expID, msg=f'Train_eval_epoch: {step+1} => ', content=r1)
    write_to_rank_file(expID=hypm.expID, step=step, row=r1)

    print(r1)
    # mdl.train()



def train(model, criterion, optimizer, train_loader, num_epochs=10, dev='cpu'):
    model.train()

    epoch_loss = []

    time_stamp()
    for epoch in range(num_epochs):
        # total_loss = 0.0
        print(f'Epoch#{epoch+1}')
        running_loss = []

        # if(epoch%10==0):
        #     train_step_eval(step=epoch, mdl=model, dev=dev)

        for i, (anchor, positive, negative, txt, idx) in enumerate(tqdm(train_loader)):

            optimizer.zero_grad()
            anchor, positive, negative = anchor.to(dev), positive.to(dev), negative.to(dev)
            # anchor, positive, negative = anchor.to(torch.float16), positive.to(torch.float16), negative.to(torch.float16)

            anchor_embedding, positive_embedding, negative_embedding = model(q = anchor, r = positive, t = txt, isTrain = True, isQuery = True)


            #---------------------------------------------og------------------------------------------------------
            
            # loss, mean_p, mean_n  = criterion(anchor_embedding, positive_embedding)
            # loss  = criterion(anchor_embedding, positive_embedding)

            #-----------------------------------------------------------------------------------------------------

            #----------------------------------------------InfoNCE_2----------------------------------------------

            if(hypm.use_neg_text):
                negative_embedding = negative_embedding.unsqueeze(0).repeat(anchor_embedding.shape[0], 1, 1) 
            else:
                # negative_embedding_combine = create_neg_keys(P=negative_embedding)
                negative_embedding_combine = create_neg_keys_2(A=anchor_embedding, P=positive_embedding)
                # negative_embedding_combine = create_neg_keys_3(A=anchor_embedding, P=positive_embedding, NN=negative_embedding)


            loss  = criterion(query= anchor_embedding, positive_key = positive_embedding, negative_keys = negative_embedding_combine)
 
            #-----------------------------------------------------------------------------------------------------



            # anchor_embedding, positive_embedding = model(q = anchor, r = positive, isTrain = True, isQuery = True)
            # _, negative_embedding = model(q = anchor, r = negative, isTrain = True, isQuery = True)
            # loss  = criterion(anchor_embedding, positive_embedding, negative_embedding)
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
            loss.backward()
            optimizer.step()
            # total_loss += loss.item()
            running_loss.append(loss.cpu().detach().numpy())
            # print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        print(f"Epoch: {epoch+1}/{num_epochs} Loss: {np.mean(running_loss)}")
        epoch_loss.append(np.mean(running_loss))

        write_to_file(expID=hypm.expID, msg=f'Loss_on_epoch:{epoch+1}=>', content=np.mean(running_loss))

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
            torch.save(model, f'model_weights/{hypm.expID}/model_tr.pth')
    
    time_stamp()
    
    return epoch_loss






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

