import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
# from models.cnn_56_bridged_unet import Resnet_upscaler as UResNet
from cnn_56_bridged_unet import Resnet_upscaler as UResNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Availability: ", device)

class ResUnetFR(nn.Module):
    def __init__(self, input_shape = (3, 224, 224), num_classes = 1000):
        super().__init__()
        # transfer learn FR with trained UResNet encoder
        self.resnet = UResNet(input_shape=(3, 224, 224))

        self.entry = self.resnet.entry
        self.enc1 = self.resnet.enc1
        self.enc2 = self.resnet.enc2
        self.enc3 = self.resnet.enc3
        self.enc4 = self.resnet.enc4
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )


    def forward(self, x):
        x1 = self.entry(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        feats = self.enc4(x4)
        # print("feats shape:", feats.shape)
        out = self.classifier(feats)
        # print("final out shape:", out.shape)
        return out
# Example
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Availability: ", device)
    if(torch.cuda.is_available()):
        print(f"GPU ID: {torch.cuda.current_device()}, {torch.cuda.get_device_name(torch.cuda.current_device())}")
    model = ResUnetFR(input_shape=(3, 224, 224), num_classes=1000).to(device)
    summary(model, (3, 224, 224))
    x = torch.randn(1, 3, 224, 224)
    y = model(x.to(device))
    print("Input:", x.shape, "Output:", y.shape)