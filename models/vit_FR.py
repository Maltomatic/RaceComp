import torch
import sys
import os
from torch import nn
import torchvision
from torchsummary import summary
from models.vit_56 import VitUpscaler as VitNet
from toolkit.ArcFacePenalty import AdditiveAngularMarginPenalty as ArcFaceLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ViTFR(nn.Module):
    def __init__(self, input_shape = (3, 224, 224), num_classes = 1000):
        super().__init__()
        # transfer learn FR with trained VitNet encoder
        self.vit = VitNet(input_shape=(3, 224, 224))

        self.entry = self.vit.entry
        self.enc1 = self.vit.enc1
        self.enc2 = self.vit.enc2
        self.enc3 = self.vit.enc3
        self.enc4 = nn.Sequential(
            self.vit.enc4,
            self.vit.out
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1216, num_classes)
        )
        
        self.arc = ArcFaceLoss(s=64.0, margin=0.15)

    def forward(self, x, label=None):
        x1 = self.entry(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        feats = self.enc4(x4)
        # print("feats shape:", feats.shape)
        out = self.classifier(feats)
        # print("final out shape:", out.shape)
        # if label is not None:
        #     out = self.arc(out, label)
        return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Availability: ", device)
    if(torch.cuda.is_available()):
        print(f"GPU ID: {torch.cuda.current_device()}, {torch.cuda.get_device_name(torch.cuda.current_device())}")
    model = ViTFR(input_shape = (3, 224, 224)).to(device)
    x = torch.randn(1, 3, 224, 224)
    y = model(x.to(device))
    print("Input:", x.shape, "Output:", y.shape)