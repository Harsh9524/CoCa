import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from modules import *
from utils import weights_init
import torch
import torch.nn as nn
import torch.nn.functional as F


import math




class ConvNet(nn.Module):
    def __init__(self, planes, num_caps, caps_size, depth):
        caps_size = 16
        super(ConvNet, self).__init__()
        channels = 3
        classes = 10
        self.num_caps = num_caps
        self.caps_size = caps_size
        self.depth = depth
        

        self.layers = nn.Sequential(
            nn.Conv2d(channels, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes*2),
            nn.ReLU(True),
            nn.Conv2d(planes*2, planes*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes*2),
            nn.ReLU(True),
            nn.Conv2d(planes*2, planes*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes*4),
            nn.ReLU(True),
            nn.Conv2d(planes*4, planes*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes*4),
            nn.ReLU(True),
            nn.Conv2d(planes*4, planes*8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes*8),
            nn.ReLU(True),
            
        )
        

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        #========= ConvCaps Layers
        for d in range(1, depth):
            self.conv_layers.append(DynamicRouting2d(num_caps, num_caps, caps_size, caps_size, kernel_size=3, stride=1, padding=1))
            nn.init.normal_(self.conv_layers[0].W, 0, 0.5)
            

        final_shape = 4

        
        self.conv_pose = nn.Conv2d(8*planes, num_caps*caps_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_pose = nn.BatchNorm2d(num_caps*caps_size)
        self.fc = DynamicRouting2d(num_caps, classes, caps_size, caps_size, kernel_size=final_shape, padding=0)
        # initialize so that output logits are in reasonable range (0.1-0.9)
        nn.init.normal_(self.fc.W, 0, 0.1)

        self.apply(weights_init)

    def forward(self, x):
        out = self.layers(x)
        out_1 = torch.flatten(out, start_dim=1)
        
        
        pose = self.bn_pose(self.conv_pose(out))

        b, c, h, w = pose.shape
        pose = pose.permute(0, 2, 3, 1).contiguous()
        pose = squash(pose.view(b, h, w, self.num_caps, self.caps_size))
        pose = pose.view(b, h, w, -1)
        pose = pose.permute(0, 3, 1, 2)

        for m in self.conv_layers:
            pose = m(pose)

        out = self.fc(pose)
        
        out = out.view(b, -1, self.caps_size)
        out = out.norm(dim=-1)


        return F.normalize(out_1, dim=-1), F.normalize(out, dim=-1)
