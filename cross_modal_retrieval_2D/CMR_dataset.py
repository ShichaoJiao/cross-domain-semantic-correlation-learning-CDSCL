from tkinter.messagebox import NO
import torch
from torch.utils.data.dataset import Dataset
import os
from torchvision import transforms as T
import glob
import numpy as np
from PIL import Image


def read_3D_data(folder_target):
    target_path = []
    target_label = []

    label = [x for x in os.listdir(folder_target) if os.path.isdir(folder_target+ '/' + x)]
    label = sorted(label)
    
    cate=[os.path.join(folder_target, x) for x in sorted(label)]
    for idx, folder in enumerate(cate):
        for path in glob.glob(folder+'/*'):
            itarget_floder = path
            #ipath_target_views = os.path.join(itarget_floder, "view")

            if os.path.isdir(itarget_floder):
                target_path.append(itarget_floder)
                target_label.append(idx)
    
    return np.array(target_path, np.str), np.array(target_label, np.int)

def read_2D_data(folder_target,extname = ".JPEG"):
    query_path = []
    query_label = []

    label = [x for x in os.listdir(folder_target) if os.path.isdir(folder_target+ '/' + x)]
    label = sorted(label)
    
    cate=[os.path.join(folder_target, x) for x in sorted(label)]
    for idx, folder in enumerate(cate):
        for ipath in glob.glob(folder+'/*'+ extname):

            if os.path.isfile(ipath):
                query_path.append(ipath)
                query_label.append(idx)
    
    return np.array(query_path, np.str), np.array(query_label, np.int)


class CMR_dataset(Dataset):
    def __init__(self,folder,num_classes = 10,split='train', data_transforms = None):
        self.folder_query = os.path.join(folder,split,"sketch")
        self.folder_target = os.path.join(folder,split,"photo")
        # self.folder_target = folder_target
        
        self.query_path,self.query_label = read_2D_data(self.folder_query,extname=".png")
        self.target_path,self.target_label = read_2D_data(self.folder_target,extname=".jpg")

        self.transform = data_transforms
        self.label_num = num_classes
        self.train_labels_set = set(range(self.label_num))

        self.target_label_to_indices = {label: np.where(self.target_label == label)[0]
                                    for label in self.train_labels_set}


    def __getitem__(self, index):
        pic_path, lab = self.query_path[index], self.query_label[index]
        pic = Image.open(pic_path)
        pic = pic.convert('RGB')
        pic = self.transform(pic)

        index_ct = np.random.choice(self.target_label_to_indices[lab])
        target_path = self.target_path[index_ct]
        target_lab = lab

        pic_target = Image.open(target_path)
        pic_target = pic_target.convert('RGB')
        pic_target = self.transform(pic_target)


        # target_views = []
        # for view_path  in glob.glob(target_path+'/*.png'):
        #     im = Image.open(view_path)
        #     im = im.convert('RGB')
        #     im = self.transform(im)
        #     target_views.append(im)
        # target_views = torch.stack(target_views)

        return pic,lab,pic_target,target_lab

    def __len__(self):
        return len(self.query_path)

class CMR_dataset_target(Dataset):
    def __init__(self,folder_target,num_classes = 10, split='test',data_transforms = None):    
    # def __init__(self,folder_target,num_classes = 10, data_transforms = None):
        # self.folder_target = folder_target
        self.folder_target = os.path.join(folder_target, split)
        self.num_classes = num_classes
        self.transform = data_transforms
        self.target_path,self.target_label = read_3D_data(self.folder_target)

        print()
        #self.target_path,self.target_label = split_data(self.target_list)
    
    def __len__(self):
        return len(self.target_path)

    def __getitem__(self, index):
        X = []
        ifolder = self.target_path[index]
        lab = self.target_label[index]
        #i = 0
        for view_path  in glob.glob(ifolder+'/*.png'):
            im = Image.open(view_path)
            im = im.convert('RGB')
            im = self.transform(im)
            X.append(im)

        return torch.stack(X),lab

class CMR_dataset_query(Dataset):    
    def __init__(self,folder_query,num_classes = 10,split='test', data_transforms = None):
        self.folder_query = os.path.join(folder_query, split)
        self.num_classes = num_classes
        self.transform = data_transforms
        
        self.query_path,self.query_label = read_2D_data(self.folder_query)
    
    def __len__(self):
        return len(self.query_path)

    def __getitem__(self, index):
        pic_path, lab = self.query_path[index], self.query_label[index]
        pic = Image.open(pic_path)
        pic = pic.convert('RGB')
        pic = self.transform(pic)

        return pic,lab


        
        






