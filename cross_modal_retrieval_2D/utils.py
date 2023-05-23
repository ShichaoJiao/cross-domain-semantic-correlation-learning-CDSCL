from pathlib import Path
from random import shuffle


def split_trainval(data_root: str, train_ratio=0.8):
    train_root = Path(data_root) 
    cates = sorted([d.stem for d in train_root.glob('*') if d.is_dir()])
    train_list, val_list = [], []
    for idx, cate in enumerate(cates):
        samples = [d for d in (train_root / cate).glob('*') if d.is_dir()]
        shuffle(samples)
        len_train = int(len(samples) * train_ratio)
        assert len_train > 0
        for _i, sample in enumerate(samples):
            if _i < len_train:
                train_list.append({'path': str(sample.absolute()),'label': idx})
            else:
                val_list.append({'path': str(sample.absolute()),'label': idx})
    return train_list, val_list

import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import torch.nn as nn

def adv_acc_compute(model_ft,test_loader):

    # model_student,model_teacher = model_ft
    # Iterate over data.
    query_num=0
    database_num=0

    # extract query feature
    for inputs, labels in test_loader["show"]:
        inputs = inputs.cuda()
        labels = labels.cuda()

        #outputs = resnet_forward(model_ft,inputs)
        outputs = model_ft(inputs,1)[0]
        
        if query_num == 0:
            query_feature = outputs
            query_labels = labels
            query_inputs = inputs.cpu()
        else:
            query_feature=torch.cat([query_feature, outputs],0)
            query_labels=torch.cat([query_labels, labels],0)
            query_inputs=torch.cat([query_inputs, inputs.cpu()],0)
        query_num = query_num + inputs.size(0)

    query_feature=query_feature.cpu().numpy()
    query_labels=query_labels.cpu().numpy()

    # extract database feature
    for inputs, labels in test_loader["photo"]:
        inputs = inputs.cuda()
        labels = labels.cuda()

        #outputs = resnet_forward(model_ft,inputs)
        outputs = model_ft(inputs,0)[0]

        if database_num == 0:
            database_feature = outputs
            database_labels = labels
            database_inputs = inputs.cpu()
        else:
            database_feature=torch.cat([database_feature, outputs],0)
            database_labels=torch.cat([database_labels, labels],0)
            database_inputs=torch.cat([database_inputs, inputs.cpu()],0)
        database_num = database_num + inputs.size(0)

    database_feature=database_feature.cpu().numpy()
    database_labels=database_labels.cpu().numpy()

    AP = np.array([])#the average precision 
    top10 = np.array([])
    top50 = np.array([])
    top100 = np.array([])
    top200 = np.array([])

    for i in range(query_num):
        q_feature = query_feature[i,:]#the ith query image feature
        q_label = query_labels[i]#the ith query image label


        rsum2_fea_norm = []

        for hh in range(len(database_feature)):
            d_feature = database_feature[hh,:]
            rsum2_fea_norm = np.append(rsum2_fea_norm, np.linalg.norm( q_feature - d_feature ))

        index = np.argsort(rsum2_fea_norm)

        true_matches = 0#the num of true_matches in retrieval for the ith query image feature
        TRUE_matches = np.sum(database_labels == q_label)#the num of ground truth for the ith query image feature

        j = 0#the jth return image for the ith query image feature
        p = np.array([])#precision array for the ith query image feature
        for h in index:
            j = j+1#the jth return image
            dt_label = database_labels[h]#the jth return image label from database
            if q_label == dt_label:#true retrieval
                true_matches = true_matches + 1
                pp = true_matches/j
                p = np.append(p,pp)
            
            if j == 10:
                t10 = true_matches/10
                top10 = np.append(top10,t10)

            elif j ==50:
                t50 = true_matches/50
                top50 = np.append(top50,t50)  
  
            elif j==100:
                t100 = true_matches/100
                top100 = np.append(top100,t100)
            
            elif j==200:
                t200 = true_matches/200
                top200 = np.append(top200,t200)

            if true_matches == TRUE_matches:# retrieval finished
                ap = np.mean(p)
                AP = np.append(AP,ap)
                break

    # for sum
    mAP = np.mean(AP)
    TOP10 = np.mean(top10)
    TOP50 = np.mean(top50)
    TOP100 = np.mean(top100)
    TOP200 = np.mean(top200)
    print('{} mAP: {:.4f} TOP10: {:.4f} TOP50: {:.4f} TOP100: {:.4f}  TOP200: {:.4f}'.format('sum', mAP, TOP10, TOP50, TOP100, TOP200))

    return mAP

