import os
import torch
import argparse
from torch.utils.data import DataLoader
from losses.Sobelloss import Fusionloss
from models.block.Drop import dropblock_step
from util.TaskFusion_dataset import Fusion_dataset
from util.common import check_dirs, init_seed, gpu_info, CosOneCycle
from fusion_model import ImageFusion


def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
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
def YCrCb2RGB(input_im):
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

def train(opt,device):
    init_seed()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    gpu_info()  
    save_path = check_dirs()

    train_dataset = Fusion_dataset('train')
    print("the training dataset is length:{}".format(train_dataset.length))
    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    model = ImageFusion(opt).cuda()
    criterion = Fusionloss(device)

    if opt.finetune:
        params = [{"params": [param for name, param in model.named_parameters()
                              if "backbone" in name], "lr": opt.learning_rate / 10}, 
                  {"params": [param for name, param in model.named_parameters()
                              if "backbone" not in name], "lr": opt.learning_rate}]  
        print("Using finetune for model")
    else:
        params = model.parameters()

    optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

    for epoch in range(opt.epochs):
        model.train()
        for it, (image_vis, image_ir, name) in enumerate(train_loader):
            model.train()
            image_vis = image_vis.cuda()
           
            image_vis_ycrcb = RGB2YCrCb(image_vis).cuda()[:,0,:,:]
            image_vis_ycrcb =image_vis_ycrcb .unsqueeze(1)
            image_ir = image_ir.cuda()

            logits = model(image_vis_ycrcb.cuda(), image_ir)
            
            optimizer.zero_grad()
          
            #--------------- fusion loss
            loss_fusion, loss_in, loss_grad =criterion(image_vis_ycrcb.cuda(), image_ir, logits)
            
            loss_total = loss_fusion
        
            loss_total.backward()

            optimizer.step()
            
            print('==Epoch:[{}],[{}]/[{}],loss_total:{},loss_in: {},loss_grad: {}'
                  .format(epoch,it,len(train_loader),loss_total.item(),loss_in.item(),loss_grad.item()))

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(model.state_dict(), save_path+'/checkpoint_epoch_'+str(epoch)+'.pt')
     

        print('An epoch finished.')



if __name__ == "__main__":
    parser = argparse.ArgumentParser('image fusion train')

    parser.add_argument("--backbone", type=str, default="GTMFusion_96")

    parser.add_argument("--neck", type=str, default="fpn+drop+aspp")
    parser.add_argument("--head", type=str, default="fcn")

    parser.add_argument("--pretrain", type=str,
                       default="") 
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
  
    parser.add_argument("--finetune", type=bool, default=True)
  

    opt = parser.parse_args()
    print("\n" + "-" * 30 + "OPT" + "-" * 30)
    print(opt)    
    device = torch.device("cuda:0")
    train(opt,device)


