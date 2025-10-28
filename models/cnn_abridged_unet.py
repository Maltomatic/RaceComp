import torch
import torch.nn as nn
import torchvision
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, ker=3, pad=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=ker, padding=pad, stride=1),
            nn.BatchNorm2d(out_ch,),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=ker, padding=pad, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class Resnet_upscaler(nn.Module):
    def __init__(self, output_size=(224,224)):
        "Model structure: take 3*112*112 tensor, upscale to 3*224*224 tensor"
        super().__init__()
        self.resnet = torchvision.models.resnet.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)

        self.entry = nn.Sequential(self.resnet.conv1,
                                  self.resnet.bn1,
                                  self.resnet.relu,
                                  self.resnet.maxpool)
        self.enc1 = self.resnet.layer1
        self.enc2 = self.resnet.layer2
        self.pad = nn.ZeroPad2d(1) #prevent image shape shrink to 7*7
        self.enc3 = self.resnet.layer3
        self.enc4 = self.resnet.layer4

        self.dec4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.conv_up4 = conv_block(2048, 1024)
        
        self.dec3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up3 = conv_block(1024, 512)
        
        self.dec2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = conv_block(256, 256)
        
        self.dec1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.conv_up1 = conv_block(64, 64)

        self.exit = nn.Conv2d(64, 3, kernel_size=3, stride=1)
        self.resize = nn.Upsample(size=output_size, mode="bilinear", align_corners=False)
    
    def forward(self, x):
        #encoder
        x1 = self.entry(x)
        # print("x1 shape:", x1.shape)
        x2 = self.enc1(x1)
        # print("x2 shape:", x2.shape)
        x3 = self.enc2(x2)
        # print("x3 shape:", x3.shape)
        x3 = self.pad(x3)
        x4 = self.enc3(x3)
        # print("x4 shape:", x4.shape)
        x5 = self.enc4(x4)
        # print("x5 shape:", x5.shape)
        # print("---- Decoder shapes ----")

        #decode with encode outputs
        d4 = self.dec4(x5)
        # print("d4 shape:", d4.shape)
        # print("x4 shape:", x4.shape)
        d4c = torch.cat([d4, x4], dim=1)
        # print("    d4 concatted shape:", d4c.shape)
        d4 = self.conv_up4(d4c)
        # print("    d4 after conv shape:", d4.shape)

        d3 = self.dec3(d4)
        # print("d3 shape:", d3.shape)
        # print("x3 shape:", x3.shape)
        d3c = torch.cat([d3, x3], dim=1)
        # print("    d3 concatted shape:", d3c.shape)
        d3 = self.conv_up3(d3c)
        # print("    d3 after conv shape:", d3.shape)

        d2 = self.dec2(d3)
        # print("d2 shape:", d2.shape)
        d2 = self.conv_up2(d2)
        # print("    d2 after conv shape:", d2.shape)
        
        d1 = self.dec1(d2)
        # print("d1 shape:", d1.shape)
        d1 = self.conv_up1(d1)
        # print("    d1 after conv shape:", d1.shape)
        
        out = self.exit(d1)
        out = self.resize(out)
        # print("final out shape:", out.shape)
        return out
# Example
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Availability: ", device)
    if(torch.cuda.is_available()):
        print(f"GPU ID: {torch.cuda.current_device()}, {torch.cuda.get_device_name(torch.cuda.current_device())}")
    model = Resnet_upscaler().to(device)
    summary(model, (3, 112, 112))
    x = torch.randn(1, 3, 112, 112)
    y = model(x.to(device))
    print("Input:", x.shape, "Output:", y.shape)