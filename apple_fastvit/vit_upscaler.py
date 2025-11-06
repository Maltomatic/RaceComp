import torch
import sys
import os
import models
from torch import nn
import torchvision
from torchsummary import summary
from timm.models import create_model
from models.modules.mobileone import reparameterize_model

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, ker=3, pad=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=ker, padding=pad, stride=1),
            nn.GELU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=ker, padding=pad, stride=1),
            nn.GELU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class up_block(nn.Module): # increase shape by factor of scale
    def __init__(self, in_ch, out_ch, scale = 2):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * (scale ** 2), kernel_size=3, padding=1, stride=1),
            nn.PixelShuffle(scale)
        )
    def forward(self, x):
        return self.up(x)

class VitUpscaler(nn.Module):
    def __init__(self, base_model = "fastvit_ma36", factor = 2, out_shape = (3, 224, 224)):
        super().__init__()
        # Load the base FastViT model
        self.encoder = create_model(base_model)
        chkp = torch.load('fastvit_ma36.pth.tar')
        state_dict = chkp['state_dict']
        self.encoder.load_state_dict(state_dict)

        if hasattr(self.encoder, "head"):
            self.encoder.head = nn.Identity()

        self.entry = nn.Sequential(
            self.encoder.patch_embed,
            nn.Identity()
        )
        self.enc1 = nn.Sequential(
            self.encoder.network[0],
            nn.Identity()
        )
        self.enc2 = nn.Sequential(
            self.encoder.network[1],
            self.encoder.network[2],
            nn.Identity()
        )
        self.enc3 = nn.Sequential(
            self.encoder.network[3],
            self.encoder.network[4],
            nn.Identity()
        )
        self.enc4 = nn.Sequential(
            self.encoder.network[5],
            self.encoder.network[6],
            nn.Identity()
        )
        self.out = nn.Sequential(
            self.encoder.network[7],
            self.encoder.conv_exp,
            nn.Identity()
        )
    
    def forward(self, x):
        y = self.entry(x)
        print("After entry: ", y.shape)
        y = self.enc1(y)
        print("After enc1: ", y.shape)
        y = self.enc2(y)
        print("After enc2: ", y.shape)
        y = self.enc3(y)
        print("After enc3: ", y.shape)
        y = self.enc4(y)
        print("After enc4: ", y.shape)
        y = self.out(y)
        print("Final: ", y.shape)
        return y

if __name__ == "__main__":
    model = VitUpscaler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model(torch.randn(1,3,112,112).to(device))
    # summary(model.to(device), (3, 112, 112))