from dropblock import LinearScheduler, DropBlock2D
from torch import nn


class DropBlock(nn.Module):
    """
    [Ghiasi et al., 2018] DropBlock: A regularization method for convolutional networks
    """
    def __init__(self, rate=0.15, size=7, step=50):
        super().__init__()

        self.drop = LinearScheduler(
            DropBlock2D(block_size=size, drop_prob=0.),
            start_value=0,
            stop_value=rate,
            nr_steps=step
        )


    def forward(self, feats: list):
        if self.training:  
            for i, feat in enumerate(feats):
                feat = self.drop(feat)
                feats[i] = feat
        return feats

    def step(self):
        self.drop.step()



def dropblock_step(model):
    """
    更新 dropblock的drop率
    """
    neck = model.module.neck if hasattr(model, "module") else model.neck
    if hasattr(neck, "drop"):
        neck.drop.step()
