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
import math
import time
from datetime import datetime
from collections import defaultdict
from torch.cuda.amp import autocast, GradScaler
from torchvision.models import vgg19, VGG19_Weights
import torch.nn.functional as F

from load import FairFaceDataset
from load import race_weighted_sampler
from load import classes, class_count, train_image_path, train_label_path, val_image_path, val_label_path

from models.cnn_resize_conv_upscaler import Resnet_Interpolate_Upscaler as InterNet
from models.cnn_unet import Resnet_upscaler as UResNet
from models.cnn_abridged_unet import Resnet_upscaler_trim as TrimResNet
from models.vit_upscaler import VitUpscaler as VitNet

from toolkit.VGGPerceptionLoss import PerceptualLossVGG19
from toolkit.debugs import denormalize_imagenet
from toolkit.criteria import psnr, ssim_simple

B = 32
C = 3
H_l = W_l = 112
H_h = W_h = 224

model_list = [
    VitNet(base_model = "fastvit_ma36", factor = 2, out_shape = (3, 224, 224)), 
    InterNet(px_shuffle=False, px_shuffle_interpolate=True, px_buffer = True, px_out=False, downsampler = False),
    TrimResNet(output_size=(224,224), px_shuffle = True),
    UResNet(px_shuffle = True)
]
model_names = ["VitNet", "InterNet", "TrimResNet", "UResNet"]

#################### configs #################### 
TRAINING = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
desc = "CNN_Unet validation info"
training_comment = "unet model testing microbatch to test GPU utilization"

model_idx = 3
Modelnet = model_list[model_idx]
model_type = model_names[model_idx]
# idx:
    # 0 - VitNet
    # 1 - InterNet
    # 2 - TrimResNet
    # 3 - UResNet

# train_list = ["All", "East Asian", "Indian", "Black", "White", "Middle Eastern", "Latino_Hispanic", "Southeast Asian"]
train_list = ["All", "East Asian", "Indian"]
epoch_stages = (2, 1, 0, 0)
use_percep = True
use_ssim = False
microbatches = 8 # 1 for no microbatching, n for n-step microbatching, max 8 recommended to avoid gradient explosion
under_represented_ratio = 0.05
tgt_race = "All"
test_stage = 2
test_epoch = 3

def imagenet_denorm(x):
    mean = x.new_tensor(IMAGENET_MEAN).view(1,-1,1,1)
    std  = x.new_tensor(IMAGENET_STD).view(1,-1,1,1)
    return x * std + mean

def accumulate_by_race(bucket, race, loss, psnr_val, ssim_val):
    b = bucket.setdefault(race, {"loss": [], "psnr": [], "ssim": []})
    b["loss"].append(loss); b["psnr"].append(psnr_val); b["ssim"].append(ssim_val)

def val(model, val_loader, use_perceptual = True):

    model = model.to(device)
    scaler = torch.amp.GradScaler(device_type, enabled=True)

    criterion = nn.L1Loss()
    perc_layers = (4, 8, 12, 16)  # conv2_2, conv3_4, conv4_4, conv5_4
    perc_weights = {4: 1.0, 8: 1.0, 12: 1.0, 16: 1.0}  # equal weighting
    lambda_perc = 0.2
    perceptual_criterion = PerceptualLossVGG19(layer_weights=perc_weights, layers=perc_layers).to(device)
    perceptual_criterion.eval()

    best_val_ssim = -1.0
    global_epoch = 0
    
    print("==Validation:==")
    with open(f"logs/val_{race}.txt", "a") as file:
        file.write("==Validation:==\n")
    v = defaultdict(float); n_val = 0; race_bucket = {}
    with torch.no_grad():
        for X_img, Y_img, labels, label_str in val_loader:
            Y_img = Y_img.to(device).float()
            X_img = X_img.to(device).float()

            pred = model(X_img)
            val_pixel = criterion(pred, Y_img)
            if use_perceptual:
                with torch.amp.autocast(device_type, enabled = True):
                    val_perc = perceptual_criterion(pred, Y_img)
                val_loss = val_pixel + lambda_perc * val_perc
                # val_loss = val_perc + lambda_perc * val_pixel
            else:
                val_loss = val_pixel

            pred_vis = imagenet_denorm(pred).clamp(0.0, 1.0)
            targ_vis = imagenet_denorm(Y_img).clamp(0.0, 1.0)
            v_psnr = psnr(pred_vis, targ_vis).mean().item()
            v_ssim = ssim_simple(pred_vis, targ_vis).mean().item()

            v["loss"] += val_loss.item()
            v["psnr"] += v_psnr
            v["ssim"] += v_ssim
            n_val += 1

            for r in label_str:
                accumulate_by_race(race_bucket, r, val_loss.item(), v_psnr, v_ssim)

    val_loss = v["loss"]/n_val
    val_psnr = v["psnr"]/n_val
    val_ssim = v["ssim"]/n_val
    print(f"           val:   loss {val_loss:.4f}  PSNR {val_psnr:.2f}  SSIM {val_ssim:.4f}")

    race_summary = {}
    for k, v in race_bucket.items():
        race_summary[k] = {m: float(np.mean(vals)) if len(vals)>0 else float('nan') for m, vals in v.items()}
    
    if race_summary:
        print("           per-race (val):",
                "  ".join([f"{k}: SSIM {vals['ssim']:.3f}, PSNR {vals['psnr']:.2f}"
                            for k, vals in race_summary.items()]))
        with open(f"logs/val_{race}.txt", "a") as file:
            file.write("per-race (val): " + "  ".join([f"{k}: SSIM {vals['ssim']:.3f}, PSNR {vals['psnr']:.2f}" for k, vals in race_summary.items()]))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    model = Modelnet.to(device)
    print("Load model from checkpoint for inference/testing")
    for race in train_list:
        ckpt_path = f"checkpoints_{model_type}/minority_{race.replace(" ", "_")}/best_stage{test_stage}_epoch{test_epoch}.pt"
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict = False)
        model.eval()
        print(f"Loaded model from {ckpt_path} for race {race} | val SSIM: {ckpt['val_ssim']:.4f}")
        with open(f"logs/val_{race}.txt", "a") as file:
            file.write(f"\nLoaded model for {race} from {ckpt_path} | val SSIM: {ckpt['val_ssim']:.4f}\n")
        model.to(device)
        val_dataset = FairFaceDataset(val_image_path, val_label_path)
        val_loader = DataLoader(val_dataset, batch_size=B, shuffle=False, num_workers=8, pin_memory=True)
        val(model, val_loader, use_perceptual = True)