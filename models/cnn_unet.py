import torch
import torch.nn as nn
import torchvision
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Availability: ", device)

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, ker=3, pad=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=ker, padding=pad, stride=1),
            # nn.BatchNorm2d(out_ch,),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=ker, padding=pad, stride=1),
            # nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class Resnet_upscaler(nn.Module):
    def __init__(self, px_shuffle = True):
        "Model structure: take 3*112*112 tensor, upscale to 3*224*224 tensor"
        super().__init__()
        self.resnet = torchvision.models.resnet.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)

        self.entry = nn.Sequential(self.resnet.conv1,
                                  self.resnet.bn1,
                                  self.resnet.relu)
        self.enc1 = nn.Sequential(self.resnet.maxpool,
                                  self.resnet.layer1)
        self.enc2 = self.resnet.layer2
        self.enc3 = self.resnet.layer3
        self.enc4 = self.resnet.layer4

        if(not px_shuffle):
            self.dec4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        else:
            self.dec4 = nn.Sequential(
                nn.Conv2d(2048, 1024*(2**2), kernel_size=3, padding=1, stride=1),
                nn.PixelShuffle(upscale_factor=2)
            )
        self.conv_up4 = conv_block(2048, 1024)
        
        if(not px_shuffle):
            self.dec3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        else:
            self.dec3 = nn.Sequential(
                nn.Conv2d(1024, 512*(2**2), kernel_size=3, padding=1, stride=1),
                nn.PixelShuffle(upscale_factor=2)
            )
        self.conv_up3 = conv_block(1024, 512)
        
        if(not px_shuffle):
            self.dec2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        else:
            self.dec2 = nn.Sequential(
                nn.Conv2d(512, 256*(2**2), kernel_size=3, padding=1, stride=1),
                nn.PixelShuffle(upscale_factor=2)
            )
        self.conv_up2 = conv_block(512, 256)
        
        if(not px_shuffle):
            self.dec1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        else:
            self.dec1 = nn.Sequential(
                nn.Conv2d(256, 128*(2**2), kernel_size=3, padding=1, stride=1),
                nn.PixelShuffle(upscale_factor=2)
            )
        self.conv_up1 = conv_block(128+64, 128)

        if(px_shuffle):
            self.dec0 = nn.Sequential(
                nn.Conv2d(128, 64*(2**2), kernel_size=3, padding=1, stride=1),
                nn.PixelShuffle(2)
            )
            self.conv_up0 = conv_block(64, 3*(2**2))
            self.up_exit = nn.PixelShuffle(2)
            self.final = nn.Conv2d(3, 3, kernel_size=3, padding = 1, stride=1)
        else:
            self.dec0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.conv_up0 = conv_block(64, 64)
            self.up_exit = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
            self.final = nn.Conv2d(64, 3, kernel_size=3, padding = 1, stride=1)

    def center_crop(self, tensor, target_size):
        _, _, h, w = tensor.size()
        _, _, th, tw = target_size
        i = (h - th) // 2
        j = (w - tw) // 2
        return tensor[:, :, i:i+th, j:j+tw]

    def forward(self, x):
        #encoder
        x1 = self.entry(x)
        # print("x1 shape:", x1.shape)
        x2 = self.enc1(x1)
        # print("x2 shape:", x2.shape)
        x3 = self.enc2(x2)
        # print("x3 shape:", x3.shape)
        x4 = self.enc3(x3)
        # print("x4 shape:", x4.shape)
        x5 = self.enc4(x4)
        # print("x5 shape:", x5.shape)

        # print("---- Decoder shapes ----")

        #decode with encode outputs
        d4 = self.dec4(x5)
        d4 = self.center_crop(d4, x4.size())
        # print("d4 shape:", d4.shape)
        # print("x4 shape:", x4.shape)
        d4 = torch.cat([d4, x4], dim=1)
        # print("    d4 concatted shape:", d4.shape)
        d4 = self.conv_up4(d4)
        # print("    d4 after conv shape:", d4.shape)

        d3 = self.dec3(d4)
        d3 = self.center_crop(d3, x3.size())
        # print("d3 shape:", d3.shape)
        # print("x3 shape:", x3.shape)
        d3 = torch.cat([d3, x3], dim=1)
        # print("    d3 concatted shape:", d3.shape)
        d3 = self.conv_up3(d3)
        # print("    d3 after conv shape:", d3.shape)

        d2 = self.dec2(d3)
        d2 = self.center_crop(d2, x2.size())
        # print("d2 shape:", d2.shape)
        # print("x2 shape:", x2.shape)
        d2 = torch.cat([d2, x2], dim=1)
        # print("    d2 concatted shape:", d2.shape)
        d2 = self.conv_up2(d2)
        # print("    d2 after conv shape:", d2.shape)
        
        d1 = self.dec1(d2)
        d1 = self.center_crop(d1, x1.size())
        # print("d1 shape:", d1.shape)
        # print("x1 shape:", x1.shape)
        d1 = torch.cat([d1, x1], dim=1)
        # print("    d1 concatted shape:", d1.shape)
        d1 = self.conv_up1(d1)
        # print("    d1 after conv shape:", d1.shape)

        d0 = self.dec0(d1)
        # print("d0 shape:", d0.shape)
        d0 = self.conv_up0(d0)
        # print("    d0 after conv shape:", d0.shape)
        
        out = self.up_exit(d0)
        out = self.final(out)
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