import torch
import torch.nn as nn
import torchvision
resnet = torchvision.models.resnet.resnet50(pretrained=True)

def conv_block(in_ch, out_ch, ker=3, pad=1):
    block = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, ker, pad),
        nn.BatchNorm2d(out_ch,),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, ker, pad),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )
    return block

class UNetResnet(nn.module):
    def __init__(self, n_classes):
        #