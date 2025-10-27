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
from collections import defaultdict
from torch.cuda.amp import autocast, GradScaler



from load import FairFaceDataset
from load import classes, class_count, train_image_path, train_label_path, val_image_path, val_label_path
from models.cnn_abridged_unet import Resnet_upscaler as TrimResNet
from models.cnn_unet import Resnet_upscaler as UResNet

B = 64
C = 3
H_l = W_l = 112
H_h = W_h = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

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

def make_param_groups(model, base_lr=3e-4, dec_mult=1.0, enc_mult=0.25):
    enc_names = {"entry","enc1","enc2","enc3","enc4"}
    enc_ids = set()
    enc_params, dec_params = [], []
    for name in enc_names:
        m = getattr(model, name, None)
        if m is None: continue
        for p in m.parameters():
            enc_params.append(p); enc_ids.add(id(p))
    for p in model.parameters():
        if id(p) not in enc_ids:
            dec_params.append(p)
    return [
        {"params": dec_params, "lr": base_lr*dec_mult},
        {"params": enc_params, "lr": base_lr*enc_mult},
    ]

def imagenet_denorm(x):
    mean = x.new_tensor(IMAGENET_MEAN).view(1,-1,1,1)
    std  = x.new_tensor(IMAGENET_STD).view(1,-1,1,1)
    return x * std + mean

def psnr(pred, target, eps=1e-8):
    mse = torch.mean((pred - target) ** 2, dim=(1,2,3)) + eps
    return 20.0 * torch.log10(1.0 / torch.sqrt(mse))

def ssim_simple(pred, target, C1=0.01**2, C2=0.03**2):
    mu_x = torch.mean(pred, dim=(2,3), keepdim=True)
    mu_y = torch.mean(target, dim=(2,3), keepdim=True)
    sigma_x = torch.var(pred, dim=(2,3), unbiased=False, keepdim=True)
    sigma_y = torch.var(target, dim=(2,3), unbiased=False, keepdim=True)
    sigma_xy = torch.mean((pred - mu_x) * (target - mu_y), dim=(2,3), keepdim=True)
    num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    return (num / (den + 1e-8)).squeeze().mean(dim=1)

def accumulate_by_race(bucket, race, loss, psnr_val, ssim_val):
    b = bucket.setdefault(race, {"loss": [], "psnr": [], "ssim": []})
    b["loss"].append(loss); b["psnr"].append(psnr_val); b["ssim"].append(ssim_val)

def train(model, 
          train_loader, 
          val_loader, 
          stages=(["enc4"], ["enc4","enc3"], ["enc4","enc3","enc2"], ["entry","enc1","enc2","enc3","enc4"]),
          epochs_per_stage=(2, 2, 2, 4),
          lr=0.003,
          out_dir="checkpoints"):
    
    os.makedirs(out_dir, exist_ok=True)
    model = model.to(device)
    scaler = GradScaler(enabled=True)

    criterion = nn.L1Loss()
    best_val_ssim = -1.0
    global_epoch = 0

    for stage_idx, layer_list in enumerate(stages):
        # Unfreeze policy
        for name in ["entry","enc1","enc2","enc3","enc4"]:
            m = getattr(model, name, None)
            if m is not None:
                for param in m.parameters():
                    param.requires_grad = False
            for name in layer_list:
                m = getattr(model, name, None)
                if m is not None:
                    for param in m.parameters():
                        param.requires_grad = True  
            # decoder always trainable
            enc_names = {"entry","enc1","enc2","enc3","enc4"}
            for name, m in model.named_children():
                if name not in enc_names:
                    for param in m.parameters():
                        param.requires_grad = True
        total_epochs = epochs_per_stage[stage_idx]
        optimizer = torch.optim.AdamW(make_param_groups(model, lr, dec_mult=1.0, enc_mult=0.25), weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs * len(train_loader)))

        print(f"\n=== Stage {stage_idx+1}/{len(stages)} | Unfrozen: {layer_list} ===")

        for e in range(total_epochs):
            global_epoch += 1
            model.train()
            tr = defaultdict(float); n_batches = 0
            for images, src, labels, label_str in train_loader:  # HR, LR
                src    = src.to(device).float()
                images = images.to(device).float()

                with autocast(enabled=True):
                    pred = model(src)
                    loss = criterion(pred, images)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                pred_vis = imagenet_denorm(pred).clamp(0.0, 1.0)
                targ_vis = imagenet_denorm(images).clamp(0.0, 1.0)
                tr["loss"] += loss.item()
                tr["psnr"] += psnr(pred_vis, targ_vis).mean().item()
                tr["ssim"] += ssim_simple(pred_vis, targ_vis).mean().item()
                n_batches += 1

            print(f"Epoch {global_epoch:03d} | train: loss {tr['loss']/n_batches:.4f}  "
                  f"PSNR {tr['psnr']/n_batches:.2f}  SSIM {tr['ssim']/n_batches:.4f}")
            
            model.eval()
            v = defaultdict(float); n_val = 0; race_bucket = {}
            with torch.no_grad():
                for images, src, labels, label_str in val_loader:
                    src    = src.to(device).float()
                    images = images.to(device).float()

                    pred = model(src)
                    val_loss = criterion(pred, images)

                    pred_vis = imagenet_denorm(pred).clamp(0.0, 1.0)
                    targ_vis = imagenet_denorm(images).clamp(0.0, 1.0)
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

            if val_ssim > best_val_ssim:
                best_val_ssim = val_ssim
                ckpt = {
                    "model": model.state_dict(),
                    "val_loss": val_loss, "val_psnr": val_psnr, "val_ssim": val_ssim,
                    "race_summary": race_summary,
                    "stage": stage_idx+1, "epoch": global_epoch
                }
                path = os.path.join(out_dir, f"best_stage{stage_idx+1}_epoch{global_epoch}.pt")
                torch.save(ckpt, path)
                print(f"Saved best checkpoint → {path} (SSIM={val_ssim:.4f})")

    # encoder_layers = {
    #     "entry": model.entry,
    #     "enc1": model.enc1,
    #     "enc2": model.enc2,
    #     "enc3": model.enc3,
    #     "enc4": model.enc4
    # }

    # TODO: gradually decrease lr and unfreeze more layers of ResNet in each stage; decoder maintains higher lr
    #Initial: only train decoder
    # for param in model.parameters():
    #     param.requires_grad = True
    # for param in model.entry.parameters(): param.requires_grad = False
    # for param in model.enc1.parameters(): param.requires_grad = False
    # for param in model.enc2.parameters(): param.requires_grad = False
    # for param in model.enc3.parameters(): param.requires_grad = False
    # for param in model.enc4.parameters(): param.requires_grad = False
    
    if __name__ == "__main__":
        train(
            model,
            train_loader,
            val_loader,
            stages=(["enc4"], ["enc4","enc3"], ["enc4","enc3","enc2"], ["entry","enc1","enc2","enc3","enc4"]),
            epochs_per_stage=(1, 1, 1, 2),   # start small for testing
            base_lr=3e-4,
            out_dir="checkpoints_unet",
            use_amp=True
        )