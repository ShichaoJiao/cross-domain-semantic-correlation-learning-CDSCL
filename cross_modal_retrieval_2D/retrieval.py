import torch
import torch.nn as nn
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm
from utils import adv_acc_compute
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

def save_data(list,save_path):
    num = len(list)
    with open(save_path, "w", newline="") as csvfile:
        #writer = csv.writer(csvfile)
        for index in range(num):
            ipath = str(list[index])
            csvfile.writelines(ipath)
            csvfile.writelines("\n")
            #writer.writerow(str(ipath))
    csvfile.close()


folder = "/media/jiao/data/jiao/datasets/Sketchy/"
save_model_path = "model/Sketchy/model.pth"


num_category = 104
batch_size = 50
num_epochs = 50
input_size = 224

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize([input_size, input_size]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

from CMR_dataset import CMR_dataset
# train_dataset = CMR_dataset(folder,num_classes = num_category,split='train',  data_transforms = data_transforms["train"])
# kwargs = {'num_workers': 16, 'pin_memory': True} if torch.cuda.is_available() else {}
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# Load Test Data
test_dataset = {x: datasets.ImageFolder(os.path.join(folder, 'test', x), data_transforms['test']) for x in ['show', 'photo']}
test_loader = {x: DataLoader(test_dataset[x], batch_size=batch_size, shuffle=False, num_workers=4) for x in ['show', 'photo']}



from CMR_model import CMR_model
model = CMR_model(num_category,flag=True)
model = model.cuda()



from utils import adv_acc_compute
model.eval()
model.load_state_dict(torch.load(save_model_path))
with torch.no_grad():
    adv_acc_compute(model,test_loader)