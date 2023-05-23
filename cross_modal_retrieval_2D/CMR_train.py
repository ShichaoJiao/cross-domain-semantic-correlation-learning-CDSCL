import torch
import torch.nn as nn
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.optim as optim
from tqdm import tqdm
from losses import grad_reverse
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
from utils import adv_acc_compute
import os

folder = "/media/jiao/data/jiao/datasets/Sketchy/"
save_model_path = "model/Sketchy/model.pth"


# train_list, val_list = split_trainval(folder_target)

# print(f'train samples: {len(train_list)}')
# print(f'val samples: {len(val_list)}')

num_category = 104
batch_size = 32
num_epochs = 50
input_size = 224

# data_transforms = {
#     'train': transforms.Compose([
#         transforms.Resize(256),
#         transforms.RandomCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor()

#     ]),
#     'test': transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor()
#     ]),
# }

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

from CMR_dataset import CMR_dataset,CMR_dataset_query,CMR_dataset_target
train_dataset = CMR_dataset(folder,num_classes = num_category,split='train',  data_transforms = data_transforms["train"])
kwargs = {'num_workers': 16, 'pin_memory': True} if torch.cuda.is_available() else {}
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# Load Test Data
test_dataset = {x: datasets.ImageFolder(os.path.join(folder, 'test', x), data_transforms['test']) for x in ['sketch', 'photo']}
test_loader = {x: DataLoader(test_dataset[x], batch_size=batch_size, shuffle=False, num_workers=4) for x in ['sketch', 'photo']}



from CMR_model import CMR_model,Discriminator
model = CMR_model(num_category,flag=True)
model = model.cuda()

model_disc = Discriminator()
model_disc = model_disc.cuda()

lr = 0.01
betas = (0.5, 0.999)

import losses
criterion_ce = nn.CrossEntropyLoss()
criterion_dis = nn.BCELoss()
criterion_sim = nn.MSELoss()
criterion_cen = losses.CrossModalCenterLoss(num_category, feat_dim=1024)


optimizer_ce = optim.SGD(list(model.parameters())+list(criterion_cen.parameters()), lr=lr)
optimizer_dis = optim.Adam(model_disc.parameters(), lr=0.001, betas=(0.5, 0.999))
scheduler_ft = lr_scheduler.StepLR(optimizer_ce, step_size=10, gamma=0.5)

best_map = 0.0

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    print('-' * 10)

    running_loss = 0.0
    running_corrects_query = 0.0
    running_corrects_target = 0.0

    model.train()
    model_disc.train()
    #for i, (query_pic,query_lab,target_pic,target_lab) in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
    for i, (query_pic,query_lab,target_pic,target_lab) in enumerate(train_loader, 0):
        query_pic = query_pic.cuda()
        query_lab = query_lab.cuda()
 
        target_pic = target_pic.cuda()
        target_lab = target_lab.cuda()

        optimizer_ce.zero_grad()
        optimizer_dis.zero_grad()

        query_feature_pub,query_predict = model(query_pic,1)
        target_feature_pub,target_predict = model(target_pic,0)
        
        dis_output_query = model_disc(query_feature_pub).view(-1)
        dis_output_target = model_disc(target_feature_pub).view(-1)

        query_label  = torch.full((query_feature_pub.size(0),), 1.0).cuda()
        target_label  = torch.full((target_feature_pub.size(0),), 0.0).cuda()

        loss_domain = criterion_dis(grad_reverse(dis_output_query,0.5),query_label) + criterion_dis(grad_reverse(dis_output_target),target_label)
        loss_ce = criterion_ce(query_predict,query_lab) * 1.5 +  criterion_ce(target_predict,target_lab) * 0.5
        loss_cen = criterion_cen(torch.cat((query_feature_pub, target_feature_pub), dim = 0), torch.cat((query_lab, target_lab), dim = 0))
        loss_sim = criterion_sim(query_feature_pub,target_feature_pub)
        
        loss = loss_ce + loss_domain * 0.1 + loss_cen * 0.01 + loss_sim * 0.01

        _, preds0 = torch.max(query_predict, 1)
        _, preds1 = torch.max(target_predict, 1)

        running_corrects_query += torch.sum(preds0 == query_lab.data)
        running_corrects_target += torch.sum(preds1 == target_lab.data)

        loss.backward()
        optimizer_ce.step()
        optimizer_dis.step()

    acc_query = running_corrects_query.double() / len(train_loader.dataset)
    acc_target = running_corrects_target.double() / len(train_loader.dataset)

    print('Train query Instance Accuracy: %f' % acc_query)
    print('Train target Instance Accuracy: %f' % acc_target)
    print()

    scheduler_ft.step()

    acc_query = 0.0
    acc_target = 0.0
    running_corrects_query = 0.0
    running_corrects_target = 0.0
    model.eval()

    with torch.no_grad():
        map_test = adv_acc_compute(model,test_loader)
        if best_map < map_test:
            best_map = map_test
            print("保存最佳模型")
            torch.save(model.state_dict(), save_model_path)



    