import torch
from torch import nn
from torchvision.models import resnet50
import numpy as np

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

class Mapper(nn.Module):
    def __init__(self):
        super(Mapper, self).__init__()
        self.mapper = nn.Linear(1,1)
        
    def forward(self, x):
        x = self.mapper(x)
        
        return x
        
class CountingNetwork(nn.Module):
    def __init__(self, output_size=1):
        super(CountingNetwork, self).__init__()

        self.backbone = FeatureExtractor()
        self.counting_part = CountingLayers(output_size)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.counting_part(x)

        return x


class CountingLayers(nn.Module):
    def __init__(self, output_size=1):
        super(CountingLayers, self).__init__()
        self.counting_part = None
        
        if output_size == 1:
            self.counting_part = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 1))
        else:
            self.counting_part = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, output_size), nn.Sigmoid())
    
    def forward(self, x):
        x = self.counting_part(x)

        return x

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
    
        model = resnet50(pretrained=True)
        layers = list(model.children())
        layers.pop()
        for l in layers:    
            self.disableBatchTrackRunnignStats(l)
     
        self.backbone = nn.Sequential(*layers)
            
    def disableBatchTrackRunnignStats(self, module):
        if type(module) == torch.nn.modules.batchnorm.BatchNorm2d:
                module.track_running_stats = True
                module.affine = True
                module.momentum = 0.99
                return
        elif len(list(module.children())) != 0:
            for child in list(module.children()):
                self.disableBatchTrackRunnignStats(child)
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1,x.size()[1])

        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(2048, 1024), 
            nn.LeakyReLU(), 
            nn.Linear(1024, 512), 
            nn.LeakyReLU(), 
            nn.Linear(512, 1))
    
    def forward(self, x):
        x = self.discriminator(x)
        return x
