import os
import torch
import random
import cv2
import numpy as np
from pathlib import Path
import matplotlib
from tqdm import tqdm
import torch.nn.functional as F

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn

from shutil import copytree, ignore_patterns




def init_seed(seed=777):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from torch.backends import cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def check_dirs():
    print("\n"+"-"*30+"Check Dirs"+"-"*30)
    if not os.path.exists('./runs'):
        os.makedirs('./runs/train')
      
    file_names = os.listdir('./runs/train')
    file_names = [int(i) for i in file_names] + [0]
    new_file_name = str(max(file_names) + 1)

    save_path = './runs/train/' + new_file_name
    os.mkdir(save_path)
    
    print("checkpoints & results are saved at: {}".format(save_path))


    return save_path

def gpu_info():
    print("\n" + "-" * 30 + "GPU Info" + "-" * 30)
    gpu_count = torch.cuda.device_count()
    x = [torch.cuda.get_device_properties(i) for i in range(gpu_count)]
    s = 'Using CUDA '
    c = 1024 ** 2  # bytes to MB
    if gpu_count > 0:
        print("Using GPU count: {}".format(torch.cuda.device_count()))
        for i in range(0, gpu_count):
            if i == 1:
                s = ' ' * len(s)
            print("%sdevice%g name='%s', memory=%dMB" % (s, i, x[i].name, x[i].total_memory / c))
    else:
        print("Using CPU !!!")

class CosOneCycle: 
    def __init__(self, optimizer, max_lr, epochs, min_lr=None, up_rate=0.3): 
        self.optimizer = optimizer

        self.max_lr = max_lr
        if min_lr is None:
            self.min_lr = max_lr / 10
        else:
            self.min_lr = min_lr
        self.final_lr = self.min_lr / 50

        self.new_lr = self.min_lr

        self.step_i = 0
        self.epochs = epochs
        self.up_rate = up_rate 
        assert up_rate < 0.5, "up_rate should be smaller than 0.5"

    def step(self):

        self.step_i += 1
        if self.step_i < (self.epochs*self.up_rate):
            self.new_lr = 0.5 * (self.max_lr - self.min_lr) * (
                        np.cos((self.step_i/(self.epochs*self.up_rate) + 1) * np.pi) + 1) + self.min_lr
        else:
            self.new_lr = 0.5 * (self.max_lr - self.final_lr) * (np.cos(
                ((self.step_i - self.epochs * self.up_rate) / (
                            self.epochs * (1 - self.up_rate))) * np.pi) + 1) + self.final_lr

        if len(self.optimizer.state_dict()['param_groups']) == 1:
            self.optimizer.param_groups[0]["lr"] = self.new_lr
        elif len(self.optimizer.state_dict()['param_groups']) == 2:  # for finetune
            self.optimizer.param_groups[0]["lr"] = self.new_lr / 10
            self.optimizer.param_groups[1]["lr"] = self.new_lr
        else:
            raise Exception('Error. You need to add a new "elif". ')


    def plot_lr(self):
        all_lr = []
        for i in range(self.epochs):
            all_lr.append(self.new_lr)
            self.step()
        fig = seaborn.lineplot(x=range(self.epochs), y=all_lr)
        fig = fig.get_figure()
        fig.savefig('./lr_schedule.jpg', dpi=200)
        self.step_i = 0
        self.new_lr = self.min_lr






