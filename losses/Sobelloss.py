#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np



class Fusionloss(nn.Module):
    def __init__(self,device):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy(device) 
        self.device=device
        
   
    def forward(self,image_vis,image_ir,generate_img):

        x_in_max=torch.max(image_vis,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        
        vis_grad=self.sobelconv(image_vis).to(self.device)
        ir_grad=self.sobelconv(image_ir).to(self.device)
        generate_img_grad=self.sobelconv(generate_img)
        
        x_grad_joint=torch.max(vis_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
       
        loss_total=loss_in+10*loss_grad  
        
        
        return loss_total,loss_in,loss_grad

class Sobelxy(nn.Module):
    def __init__(self,device):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(device)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

if __name__ == '__main__':
    pass

