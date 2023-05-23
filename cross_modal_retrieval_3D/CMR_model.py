import imp
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from torchvision import models
import numpy as np
from torchvision import models

class FeatureNet(nn.Module):
    def __init__(self, pretrained=False):
        super(FeatureNet, self).__init__()

        self.base_model = models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])


    def forward(self, x):
        # feature maps
        x = self.features(x)
        # flatten
        x = x.view(x.size(0), -1)
        return x


class CMR_model(nn.Module):
    def __init__(self, num_category,n_view,flag=False):
        super(CMR_model, self).__init__()

        self.n_view = n_view

        self.FeatureNet_query = FeatureNet(pretrained = flag)
        self.FeatureNet_target = FeatureNet(pretrained = flag)

        self.common_space  = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=1024, bias=True),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
           nn.Linear(in_features=1024, out_features=num_category, bias=True)
        )

    def forward(self,x,is_query):
        if is_query == 1:
            feature_x = self.FeatureNet_query(x)
        else:
            view_batch = x
            assert view_batch.size(1) == self.n_view
            view_batch = view_batch.view(-1, view_batch.size(2), view_batch.size(3), view_batch.size(4))
            view_fts = self.FeatureNet_target(view_batch)
            view_fts = view_fts.view(view_fts.size(0), -1)
            local_view_fts = view_fts.view(-1, self.n_view, view_fts.size(-1))
            feature_x, _ = local_view_fts.max(dim=1)

        feature_x = self.common_space(feature_x)
        output_x = self.classifier(feature_x)
        return feature_x,output_x

        
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(1024,128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)

