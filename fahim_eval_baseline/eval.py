import torch
from tqdm import tqdm
import time
import copy
import numpy as np
from torch.cuda.amp import autocast
import torch.nn.functional as F
from attributes import Configuration as hypm
import os

from helper_func import save_tensor, idsToDist
from torch.amp import autocast

def predict(model, dataloader, verbose=True, dev=torch.device('cpu'), normalize_features=True, isQuery=True):
    

    model.eval()

    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    query_features_list = []
    ref_feature_list = []
    
    ids_list = []
    with torch.no_grad():
        
        # for img, ids, txt in bar:
        for anchor, positive, negative, txt, idx in bar:
        
            ids_list.append(idx)
            # img = img.to(dev)
            anchor, positive, negative = anchor.to(dev), positive.to(dev), negative.to(dev)

            # if(isQuery):
            #     img_feature = model(q = img, r = img, t=txt, isTrain = False, isQuery = True)
            # else:
            #     img_feature = model(q = img, r = img, t=txt, isTrain = False, isQuery = False)
            with autocast(device_type='cuda', enabled=hypm.use_mixed_precision):
                query_feature,  ref_feature, _ = model(q = anchor, r = positive, t=txt, isTrain = False, isQuery = True)


            # if(dev !='cpu'):    
            #     with autocast():
            #         img = img.to(dev)
            #         if(isQuery):
            #             img_feature = model(q = img, r = img, isTrain = False, isQuery = True)
            #         else:
            #             img_feature = model(q = img, r = img, isTrain = False, isQuery = False)
            # else:
            #     img = img.to(dev)
            #     if(isQuery):
            #         img_feature = model(q = img, r = img, isTrain = False, isQuery = True)
            #     else:
            #         img_feature = model(q = img, r = img, isTrain = False, isQuery = False)
        
            # normalize is calculated in fp32
            # if normalize_features:
            #     img_feature = F.normalize(img_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            # img_features_list.append(img_feature.to(torch.float32))
            query_features_list.append(query_feature)
            ref_feature_list.append(ref_feature)


      
        # keep Features on GPU
        # img_features = torch.cat(img_features_list, dim=0) 

        query_features = torch.cat(query_features_list, dim=0) 
        ref_features = torch.cat(ref_feature_list, dim=0) 

        ids_list = torch.cat(ids_list, dim=0).to(dev)

        # print(f'img_feature: {img_features.shape}')
        # print(f'ids: {ids_list.shape}')
        
    if verbose:
        bar.close()
        
    return query_features,  ref_features, ids_list

# def predict2(model, dev=torch.device('cpu'), normalize_features=True, isQuery=True):
    
#     model.eval()

#     # wait before starting progress bar
#     time.sleep(0.1)
        
#     query_features_list = []
#     ref_feature_list = []
    
#     ids_list = []
#     with torch.no_grad():
#         for anchor, positive, negative, txt, idx in :
#             ids_list.append(idx)
#             anchor, positive, negative = anchor.to(dev), positive.to(dev), negative.to(dev)

#             with autocast(device_type='cuda', enabled=hypm.use_mixed_precision):
#                 query_feature,  ref_feature, _ = model(q = anchor, r = positive, t=txt, isTrain = False, isQuery = True)

#             query_features_list.append(query_feature)
#             ref_feature_list.append(ref_feature)


      


#         query_features = torch.cat(query_features_list, dim=0) 
#         ref_features = torch.cat(ref_feature_list, dim=0) 

#         ids_list = torch.cat(ids_list, dim=0).to(dev)


        
#     return query_features,  ref_features, ids_list

def calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=1000, ranks=[1,5,10]):

    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)
    
    steps = Q // step_size + 1
    
    
    query_labels_np = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    
    
    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i
    
    
    similarity = []
    
    for i in range(steps):
        
        start = step_size * i
        
        end = start + step_size
          
        sim_tmp = query_features[start:end] @ reference_features.T
        
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)
 
    

    topk.append(R//100)
    
    results = np.zeros([len(topk)])
    
    
    bar = tqdm(range(Q))
    
    for i in bar:
        # print(f'i={i}, query_labels_np={query_labels_np[i]}, ref2index={ref2index[query_labels_np[i]]}')
        # similiarity value of gt reference
        gt_sim = similarity[i, ref2index[query_labels_np[i]]]
        
        # number of references with higher similiarity as gt
        higher_sim = similarity[i,:] > gt_sim
        
         
        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
                        
        
    results = results/ Q * 100.
 
    
    bar.close()
    
    # wait to close pbar
    time.sleep(0.1)
    
    string = []
    for i in range(len(topk)-1):
        
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
        
    string.append('Recall@top1: {:.4f}'.format(results[-1]))            
        
    print(' - '.join(string)) 

    return results


def accuracy(query_features, reference_features, query_labels, topk=[1,5,10], tv_all_reference_features = None, q_item=-1):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # print(f'query labels {query_labels}')
    ts = time.time()
    N = query_features.shape[0]
    M = reference_features.shape[0]
    topk.append(N//100)
    results = np.zeros([len(topk)])
    # for CVUSA, CVACT
    query_features = query_features.cpu()
    reference_features = reference_features.cpu()
    query_labels = query_labels.cpu()

    if N < 80000:
        query_features_norm = np.sqrt(np.sum((query_features**2).numpy(), axis=1, keepdims=True))
        reference_features_norm = np.sqrt(np.sum((reference_features ** 2).numpy(), axis=1, keepdims=True))
        similarity = np.matmul(query_features/query_features_norm, (reference_features/reference_features_norm).T)
        similarity = similarity.numpy()
        # print(similarity.shape)
        # save_tensor(var_name='similarity', var=similarity)
        #----------------------------------------------------------------------
        # for i in range(N):
        #     # ranking = np.sum((similarity[i,:]>similarity[i,query_labels[i]])*1.)
        #     ranking = np.sum((similarity[i,:]>similarity[i,i])*1.)
        #     # print(ranking)

        #     for j, k in enumerate(topk):
        #         if ranking < k:
        #             results[j] += 1.
        #         # print(f'k: {k} == results: {results}')

        #----------------------------------------------------------------------
        if(q_item>-1):
            ranking = np.sum((similarity[q_item,:]>similarity[q_item,q_item])*1.)

            for j, k in enumerate(topk):
                if ranking < k:
                    results[j] += 1.
            return results
        else:
            for i in range(N):
                # ranking = np.sum((similarity[i,:]>similarity[i,query_labels[i]])*1.)
                ranking = np.sum((similarity[i,:]>similarity[i,i])*1.)
                # print(ranking)

                for j, k in enumerate(topk):
                    if ranking < k:
                        results[j] += 1.
                # print(f'k: {k} == results: {results}')
        # -----------------------------------Top 10 Retrieval------------------------------------
   
        # info_filepath = f'info/info_{hypm.expID}.txt'
        # for i in range(N):
        #     ranking = np.sum((similarity[i,:]>similarity[i,i])*1.)
        #     if ranking<10:
        #         better_than = (similarity[i,:]>similarity[i,i])*1.
        #         indices_of_ones = [index for index, value in enumerate(better_than) if value == 1]
        #         with open(info_filepath, 'a') as file:
        #             file.write(f'\ni: {i}--->')
        #             file.write(f'{indices_of_ones}\n')
        #         # print(f'i: {i}')
        #         # print(indices_of_ones)


        # ------------------------------Meter level ACC-----------------------------------------

        # info_filepath = f'info/info_{hypm.expID}.txt'
        # for i in range(N):
        #     # close1, close2 = find_two_closest_indices(similarity[i,:], similarity[i,i])
        #     ranking = np.sum((similarity[i,:]>similarity[i,i])*1.)
        #     distances=[-1]
        #     if ranking<topk[-1]:
        #         better_than = (similarity[i,:]>similarity[i,i])*1.
        #         indices_of_ones = [index for index, value in enumerate(better_than) if value == 1]

        #         distances = [idsToDist(id_a=i, id_b=idx, ll_csv=hypm.latlong_csv) for idx in indices_of_ones]

        #         # better_than = [num for i, num in enumerate(Ar) if num>X]

        #     # dist = idsToDist(id_a=i, id_b=close2, ll_csv=hypm.latlong_csv)
        #     with open(info_filepath, 'a') as file:
        #         file.write(f'\ni: {i}--->')
        #         file.write(f'{indices_of_ones}----->')
        #         file.write(f'{sorted(distances)}\n')
        #         # print(f'i: {i}')
        #         # print(indices_of_ones)


        # ------------------------------Meter level ACC v2-----------------------------------------
        # tv_all_reference_features = tv_all_reference_features.cpu()
        # # print(tv_all_reference_features.shape)
        # tv_reference_features_norm = np.sqrt(np.sum((tv_all_reference_features ** 2).numpy(), axis=1, keepdims=True))
        # tv_similarity = np.matmul(query_features/query_features_norm, (tv_all_reference_features/tv_reference_features_norm).T)
        # tv_similarity = tv_similarity.numpy()
        # print(tv_similarity.shape)

        # info_filepath = f'info/info_{hypm.expID}.txt'
        # for i in range(N):
        #     max_sim = [-1,0]

        #     for j in range(tv_similarity.shape[1]):
        #         if(tv_similarity[i,j]>similarity[i,i]):
        #             if tv_similarity[i,j]>max_sim[1]:
        #                 max_sim = [j, tv_similarity[i,j]]
            
        #     if(max_sim[0]==-1 and max_sim[1]==0):
        #         dist = 0
        #     else:
        #         dist = idsToDist(id_a=i, id_b=max_sim[0], ll_csv=hypm.latlong_csv)
            
        #     with open(info_filepath, 'a') as file:
        #         file.write(f'\ni: {i}--->')
        #         file.write(f'[{dist}]\n')

        # ******************************************************************************************************************

        #     # close1, close2 = find_two_closest_indices(similarity[i,:], similarity[i,i])
            # ranking = np.sum((tv_similarity[i,:]>similarity[i,i])*1.)
            # distances=[-1]
            # if ranking<2:
            # better_than = (tv_similarity[i,:]>similarity[i,i])*1.
            # indices_of_ones = [index for index, value in enumerate(better_than) if value == 1]
            # print(better_than)

                # distances = [idsToDist(id_a=i, id_b=idx, ll_csv=hypm.latlong_csv) for idx in indices_of_ones]

        #         # better_than = [num for i, num in enumerate(Ar) if num>X]

        #     # dist = idsToDist(id_a=i, id_b=close2, ll_csv=hypm.latlong_csv)
        #     with open(info_filepath, 'a') as file:
        #         file.write(f'\ni: {i}--->')
        #         file.write(f'{indices_of_ones}----->')
        #         file.write(f'{sorted(distances)}\n')
                # print(f'i: {i}')
                # print(indices_of_ones)
        # ---------------------------------------------------------------------------------------------
        
                  
    else:
        # split the queries if the matrix is too large, e.g. VIGOR
        assert N % 4 == 0
        N_4 = N // 4
        for split in range(4):
            query_features_i = query_features[(split*N_4):((split+1)*N_4), :]
            query_labels_i = query_labels[(split*N_4):((split+1)*N_4)]
            query_features_norm = np.sqrt(np.sum(query_features_i ** 2, axis=1, keepdims=True))
            reference_features_norm = np.sqrt(np.sum(reference_features ** 2, axis=1, keepdims=True))
            similarity = np.matmul(query_features_i / query_features_norm,
                                   (reference_features / reference_features_norm).transpose())
            for i in range(query_features_i.shape[0]):
                ranking = np.sum((similarity[i, :] > similarity[i, query_labels_i[i]])*1.)
                for j, k in enumerate(topk):
                    if ranking < k:
                        results[j] += 1.

    results = results/ query_features.shape[0] * 100.
    print('Percentage-top1:{}, top5:{}, top10:{}, top1%:{}, time:{}'.format(results[0], results[1], results[2], results[-1], time.time() - ts))
    return results





def find_two_closest_indices(Ar, X):
    """
    Find the indices of the closest and second closest numbers to X in the list Ar.

    Parameters:
    Ar : list of float or int : The list of numbers
    X : float or int : The target number

    Returns:
    tuple : (int, int) : Indices of the closest and second closest numbers to X in Ar
    """
    # Create a list of (index, absolute difference) pairs
    differences = [num for i, num in enumerate(Ar) if num>X]
    distances = [num for i, num in enumerate(Ar)]

    # Sort by absolute difference
    # differences.sort(key=lambda pair: pair[1])
    # Return the indices of the closest and second closest
    return differences[0][0], differences[1][0]