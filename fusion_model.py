import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.head.FCN import FCNHead
from models.neck.FPN import FPNNeck
from collections import OrderedDict
from typing import Dict
from GTM import model

   
mtc= model()
class ImageFusion(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.inplanes = int(re.sub(r"\D", "", opt.backbone.split("_")[-1]))
  
        self._create_backbone(opt.backbone)
        self._create_neck(opt.neck)
        self._create_heads(opt.head)

        if opt.pretrain.endswith(".pt"):
            self._init_weight(opt.pretrain)   
 

    def forward(self, xa, xb):
        _, _, h_input, w_input = xa.shape
        assert xa.shape == xb.shape, "The two images are not the same size, please check it."

        fa1, fa2, fa3, fa4 = self.backboneA(xa)  
        fb1, fb2, fb3, fb4 = self.backboneB(xb)


        ms_feats = fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4  

        fusion = self.neck(ms_feats)

        out = self.head_forward(ms_feats, fusion, out_size=(h_input, w_input))

        return out

    def head_forward(self, ms_feats, fusion, out_size):

        out = F.interpolate(self.head(fusion), size=out_size, mode='bilinear', align_corners=True)
      

        return out
    
    def _init_weight(self, pretrain=''): 
        for m in self.modules():
            if isinstance(m, nn.Conv2d): 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if pretrain.endswith('.pt'):
            pretrained_dict = torch.load(pretrain)
            if isinstance(pretrained_dict, nn.DataParallel):
                pretrained_dict = pretrained_dict.module.state_dict()
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(OrderedDict(model_dict), strict=True)
            print("=> Imagefusion load {}/{} items from: {}".format(len(pretrained_dict),
                                                                    len(model_dict), pretrain))

    def _create_backbone(self, backbone):
            
        if 'mtc' in backbone:
            self.backboneA = mtc
            self.backboneB = mtc
        else:
            raise Exception('Not Implemented yet: {}'.format(backbone))

    def _create_neck(self, neck):
        if 'fpn' in neck:
            self.neck = FPNNeck(self.inplanes, neck)

    def _select_head(self, head):
        if head == 'fcn':
            return FCNHead(self.inplanes, 1)

    def _create_heads(self, head):
        self.head = self._select_head(head)
