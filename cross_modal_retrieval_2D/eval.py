import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import torch.optim as optim
from CMR_model import CMR_model
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
import csv
import glob
from PIL import Image
from CMR_dataset import CMR_dataset_query,CMR_dataset_target
import scipy.io as sio
from utils import adv_acc_compute

num_category = 40
batch_size = 8

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()

    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]),
}


# folder_query = "/home/jiao/workspace/dataset/MI3OR/image"
# folder_target = "/home/jiao/workspace/dataset/MI3OR/view"
# save_model_path = "model/MI3OR.pth"

folder_query = "/home/jiao/workspace/dataset/IM2MN/image"
folder_target = "/home/jiao/workspace/dataset/IM2MN/view"
save_model_path = "model/IN2MN.pth"

test_dataset_query = CMR_dataset_query(folder_query,num_classes = num_category,split='test', data_transforms = data_transforms["test"])
test_dataset_target = CMR_dataset_target(folder_target,num_classes = num_category, split='test',data_transforms = data_transforms["test"])

kwargs = {'num_workers': 16, 'pin_memory': True} if torch.cuda.is_available() else {}
test_loader_query = DataLoader(test_dataset_query, batch_size=batch_size, shuffle=False, **kwargs)
test_loader_target = DataLoader(test_dataset_target, batch_size=batch_size, shuffle=False, **kwargs)

model = CMR_model(num_category,12,flag=True)
model = model.cuda()
model.load_state_dict(torch.load(save_model_path))

query_feature = []
query_label = []

target_feature = []
target_label = []


model.eval()
with torch.no_grad():
    adv_acc_compute(model,test_loader_query,test_loader_target)
    # for i, (pic,lab) in tqdm(enumerate(test_loader_query, 0), total=len(test_loader_query), smoothing=0.9):
    #     pic = pic.cuda()
    #     lab = lab.cuda()
    #     features = model(pic,1)[1]

    #     query_feature.extend(features.tolist())
    #     query_label.extend(lab.tolist())

    # for i, (pic,lab) in tqdm(enumerate(test_loader_target, 0), total=len(test_loader_target), smoothing=0.9):
    #         # for i, (pic,lab) in enumerate(test_loader_target, 0):
    #     pic = pic.cuda()
    #     lab = lab.cuda()
    #     features = model(pic,0)[1]

    #     target_feature.extend(features.tolist())
    #     target_label.extend(lab.tolist())


    # sio.savemat('MDI3D/cmcl_Results_features_labels.mat', {'source_feature':query_feature, 'source_label':query_label, \
    # 'target_feature':target_feature, 'target_label':target_label}) 
    # print('Finished Extracting Features! Source: {}, Target: {}'.format(len(query_label), len(target_label)))