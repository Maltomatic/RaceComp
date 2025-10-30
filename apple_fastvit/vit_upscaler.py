import torch
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
        if hasattr(self.encoder, "head"):
            self.encoder.head = nn.Identity()
        self.exposed_features = {}
        


# To Train from scratch/fine-tuning
model = create_model("fastvit_ma36")
# ... train ...

# Load unfused pre-trained checkpoint for fine-tuning
# or for downstream task training like detection/segmentation
checkpoint = torch.load('fastvit_ma36.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# summary(model.to(device), (3, 112, 112))

print("Model Structure:")
print(model)
# ... train ...


# # For inference
# model.eval()      
# model_inf = reparameterize_model(model)
# # Use model_inf at test-time