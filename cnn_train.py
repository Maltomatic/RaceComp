import pandas as pd
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image

from load import FairFaceDataset
from load import classes, class_count, train_image_path, train_label_path, val_image_path, val_label_path
from models.cnn_abridged_unet import Resnet_upscaler as TrimResNet
from models.cnn_unet import Resnet_upscaler as UResNet

B = 64
C = 3
H_l = W_l = 112
H_h = W_h = 224

train_dataset = FairFaceDataset(train_image_path, train_label_path)
train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True)

val_dataset = FairFaceDataset(val_image_path, val_label_path)
val_loader = DataLoader(val_dataset, batch_size=B, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

def denormalize_imagenet(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(-1, 1, 1)
    return ((tensor * std).add(mean)*255).clamp(0, 255).byte()

def show_batch(lr_batch, hr_batch, preds_batch, num_samples=4):
    lr_batch = denormalize_imagenet(lr_batch.cpu())
    hr_batch = denormalize_imagenet(hr_batch.cpu())
    preds_batch = denormalize_imagenet(preds_batch.cpu())

    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
    for i in range(num_samples):
        axes[i, 0].imshow(lr_batch[i].permute(1, 2, 0))
        axes[i, 0].set_title("Low-Res Input")
        axes[i, 1].imshow(preds_batch[i].permute(1, 2, 0))
        axes[i, 1].set_title("Model Output")
        axes[i, 2].imshow(hr_batch[i].permute(1, 2, 0))
        axes[i, 2].set_title("High-Res Target")
        for j in range(3):
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.show()

print("Number of training samples: ", len(train_dataset))
print("Number of validation samples: ", len(val_dataset))

for images, src, labels, label_str in train_loader:
    print("Batch of testing images shape: ", images.shape)
    print("Batch of source images shape: ", src.shape)
    print("Batch of labels shape: ", labels.shape)
    print("Batch of label strings: ", len(label_str))
    break

print("Half-Connected U-ResNet Model Summary:")
print(summary(TrimResNet().to(device), (3,112,112)))

print("Full U-ResNet Model Summary:")
print(summary(UResNet().to(device), (3,112,112)))

def train(model, train_loader, val_loader, epochs=200, lr=0.003):
    model = model.to(device)
    criterion = nn.MSELoss() # need to replace with something better; L1 & L2 loss are blurry
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    stages = [
        ["enc4"],                # decoder + enc4
        ["enc4", "enc3"],      # add enc3
        ["enc4", "enc3", "enc2"], # add enc2
        ["enc4", "enc3", "enc2", "enc1", "entry"] # full fine-tune
    ]

    encoder_layers = {
        "entry": model.entry,
        "enc1": model.enc1,
        "enc2": model.enc2,
        "enc3": model.enc3,
        "enc4": model.enc4
    }

    # TODO: gradually decrease lr and unfreeze more layers of ResNet in each stage; decoder maintains higher lr
    #Initial: only train decoder
    for param in model.parameters():
        param.requires_grad = True
    for param in model.entry.parameters(): param.requires_grad = False
    for param in model.enc1.parameters(): param.requires_grad = False
    for param in model.enc2.parameters(): param.requires_grad = False
    for param in model.enc3.parameters(): param.requires_grad = False
    for param in model.enc4.parameters(): param.requires_grad = False
    