import torch
import models
from torch import nn
import torchvision
from torchsummary import summary
from timm.models import create_model
from models.modules.mobileone import reparameterize_model

# To Train from scratch/fine-tuning
model = create_model("fastvit_ma36")
# ... train ...

# Load unfused pre-trained checkpoint for fine-tuning
# or for downstream task training like detection/segmentation
checkpoint = torch.load('fastvit_ma36.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summary(model.to(device), (3, 112, 112))

print("Model Structure:")
print(model)
# ... train ...
# print(type(model.patch_embed))
# print(type(model.gap))


# # For inference
# model.eval()      
# model_inf = reparameterize_model(model)
# # Use model_inf at test-time