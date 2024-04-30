
# coding:utf-8
import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from util.TaskFusion_dataset import Fusion_dataset
from tqdm import tqdm
from torch.autograd import Variable
from PIL import Image
import cv2
from fusion_model import ImageFusion

def main(opt): 
    fusion_model_path=opt.fusion_model_path
    fusionmodel= ImageFusion(opt).cuda()
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    if args.gpu >= 0:
        fusionmodel.to(device)
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    print('fusionmodel load done!')
    ir_path = opt.ir_path
    vi_path = opt.vi_path
    test_dataset = Fusion_dataset('val', ir_path=ir_path, vi_path=vi_path)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=64,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        fusionmodel.eval()
        for it, (images_vis, images_ir,name) in enumerate(test_loader):
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)
            if args.gpu >= 0:
                images_vis = images_vis.to(device)
                images_ir = images_ir.to(device)
            image_vis_ycrcb = RGB2YCrCb(images_vis).to(device)
            
            images_vis_ycrcb =image_vis_ycrcb[:,0,:,:] .unsqueeze(1)
            logits = fusionmodel(images_vis_ycrcb, images_ir)
            fusion_ycrcb = torch.cat(
                (logits, image_vis_ycrcb[:, 1:2, :, :], image_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )
            fusion_image = YCrCb2RGB(fusion_ycrcb)
         
            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
            fused_image = fusion_image.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                np.max(fused_image) - np.min(fused_image)
            )

            fused_image = np.uint8(255.0 * fused_image)

            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                image = Image.fromarray(image)
                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)
                print('Fusion {} Sucessfully!'.format(save_path))

def YCrCb2RGB(input_im):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(device)
    temp = (im_flat + bias).mm(mat).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def RGB2YCrCb(input_im):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='GTMFusion')
    parser.add_argument("--backbone", type=str, default="mtc_96")
    parser.add_argument("--neck", type=str, default="fpn+aspp+drop")
    parser.add_argument("--head", type=str, default="fcn")

    parser.add_argument("--pretrain", type=str,
                       default="")  
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    parser.add_argument('--fusion_model_path', type=str, default=
                        './checkpoint/checkpoint_epoch.pt')   
    parser.add_argument('--ir_path', type=str, default=
                        '')
    parser.add_argument('--vi_path', type=str, default=
                        '') 
    args = parser.parse_args()
  
    fused_dir = './Fusion_results'
    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    print('| testing %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    main(args)

