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
from torchvision.models import vgg19, VGG19_Weights
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler


from load import FairFaceDataset
from load import classes, class_count, train_image_path, train_label_path, val_image_path, val_label_path
from models.cnn_abridged_unet import Resnet_upscaler as TrimResNet
from models.cnn_unet import Resnet_upscaler as UResNet

B = 64
C = 3
H_l = W_l = 112
H_h = W_h = 224

TRAINING = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = "cuda" if torch.cuda.is_available() else "cpu"

torch.autograd.set_detect_anomaly(True)

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

# print("Half-Connected U-ResNet Model Summary:")
# print(summary(TrimResNet().to(device), (3,112,112)))

# print("Full U-ResNet Model Summary:")
# print(summary(UResNet().to(device), (3,112,112)))

def make_param_groups(model, base_lr=3e-4, dec_mult=1.0, enc_mult=0.5):
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

class VGG19FeatureExtractor(nn.Module):
    def __init__(self, layers=(4, 8, 12, 16),  # conv2_2, conv3_4, conv4_4, conv5_4 by conv-count index
                 weights=VGG19_Weights.IMAGENET1K_V1):
        super().__init__()
        vgg = vgg19(weights=weights).features
        # freeze
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.eval()
        self.capture_layers = set(layers)

    @torch.no_grad()
    def _sanity_check(self, x):
        return x

    def forward(self, x):
        feats = []
        conv_idx = 0
        for m in self.vgg:
            if isinstance(m, nn.Conv2d):
                conv_idx += 1
                x = m(x)
                if conv_idx in self.capture_layers:
                    feats.append(x.clone())
            else:
                x = m(x)
        return feats


class PerceptualLossVGG19(nn.Module):
    def __init__(self, layer_weights=None, layers=(4, 8, 12, 16)):
        super().__init__()
        self.feat_net = VGG19FeatureExtractor(layers=layers)
        if layer_weights is None:
            layer_weights = {l: 1.0 for l in layers}
        self.layer_weights = layer_weights
        self.layers = layers

    def forward(self, pred, target):
        with torch.cuda.amp.autocast(False): 
            pred_f   = self.feat_net(pred.float())
            target_f = self.feat_net(target.float())
        loss = 0.0
        for l, Fp, Ft in zip(self.layers, pred_f, target_f):
            w = self.layer_weights.get(l, 1.0)
            loss = loss + w * F.l1_loss(Fp, Ft, reduction='mean')
        return loss

def train(model, 
          train_loader, 
          val_loader, 
          stages=(["enc4"], ["enc4","enc3"], ["enc4","enc3","enc2"], ["entry","enc1","enc2","enc3","enc4"]),
          epochs_per_stage=(2, 2, 2, 4),
          lr=0.003,
          out_dir="checkpoints"):
    
    os.makedirs(out_dir, exist_ok=True)
    model = model.to(device)
    scaler = torch.amp.GradScaler(device_type, enabled=True)

    criterion = nn.L1Loss()
    use_perceptual = True
    perc_layers = (4, 8, 12, 16)  # conv2_2, conv3_4, conv4_4, conv5_4
    perc_weights = {4: 1.0, 8: 1.0, 12: 1.0, 16: 1.0}  # equal weighting
    lambda_perc = 0.01  # typical ESRGAN-scale; tune 0.005~0.05
    perceptual_criterion = PerceptualLossVGG19(layer_weights=perc_weights, layers=perc_layers).to(device)
    perceptual_criterion.eval()

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
            print(f"\n--- Epoch {e+1}/{total_epochs} ---")
            global_epoch += 1
            with open("train_log.txt", "a") as file:
                file.write(f"\n--- Epoch {global_epoch} ---\n")
            model.train()
            tr = defaultdict(float); n_batches = 0
            for X_img, Y_img, labels, label_str in train_loader:  # LR, HR
                if(n_batches % 300 == 1):
                    print(f"Training batch {n_batches+1}/{len(train_loader)}")
                    print(f"Batch {n_batches:03d} | train: loss {tr['loss']/n_batches:.4f}  "
                        f"PSNR {tr['psnr']/n_batches:.2f}  SSIM {tr['ssim']/n_batches:.4f}")
                    with open("train_log.txt", "a") as file:
                        file.write(f"Batch {n_batches:03d} | train: loss {tr['loss']/n_batches:.4f}  "
                                f"PSNR {tr['psnr']/n_batches:.2f}  SSIM {tr['ssim']/n_batches:.4f}\n")
                Y_img = Y_img.to(device).float()
                X_img = X_img.to(device).float()

                with torch.amp.autocast(device_type, enabled=True):
                    # print("Debug: Input shapes:", X_img.shape, Y_img.shape)
                    pred = model(X_img)
                    pixel_loss = criterion(pred, Y_img)
                    if use_perceptual:
                        with torch.cuda.amp.autocast(False):
                            perc_loss = perceptual_criterion(pred, Y_img)
                        loss = pixel_loss + lambda_perc * perc_loss
                    else:
                        loss = pixel_loss
                    
                    if n_batches % 300 == 1 and use_perceptual:
                        print(f"pixel={pixel_loss.item():.4f}  perc={perc_loss.item():.4f}  tot={loss.item():.4f}")

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                pred_vis = imagenet_denorm(pred).clamp(0.0, 1.0)
                targ_vis = imagenet_denorm(Y_img).clamp(0.0, 1.0)
                tr["loss"] += loss.item()
                tr["psnr"] += psnr(pred_vis, targ_vis).mean().item()
                tr["ssim"] += ssim_simple(pred_vis, targ_vis).mean().item()
                n_batches += 1

            print(f"Epoch {global_epoch:03d} | train: loss {tr['loss']/n_batches:.4f}  "
                  f"PSNR {tr['psnr']/n_batches:.2f}  SSIM {tr['ssim']/n_batches:.4f}")
            with open("train_log.txt", "a") as file:
                file.write(f"Epoch {global_epoch:03d} | train: loss {tr['loss']/n_batches:.4f}  "
                           f"PSNR {tr['psnr']/n_batches:.2f}  SSIM {tr['ssim']/n_batches:.4f}\n")
            
            model.eval()
            print("==Validation:==")
            with open("train_log.txt", "a") as file:
                file.write("==Validation:==\n")
            v = defaultdict(float); n_val = 0; race_bucket = {}
            with torch.no_grad():
                for X_img, Y_img, labels, label_str in val_loader:
                    Y_img = Y_img.to(device).float()
                    X_img = X_img.to(device).float()

                    pred = model(X_img)
                    val_pixel = criterion(pred, Y_img)
                    if use_perceptual:
                        with torch.cuda.amp.autocast(False):
                            val_perc = perceptual_criterion(pred, Y_img)
                        val_loss = val_pixel + lambda_perc * val_perc
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
                with open("train_log.txt", "a") as file:
                    file.write("per-race (val): " + "  ".join([f"{k}: SSIM {vals['ssim']:.3f}, PSNR {vals['psnr']:.2f}" for k, vals in race_summary.items()]))

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
        
        del optimizer
        del scheduler



def race_weighted_sampler(dataset, race_weights, num_samples, seed=42):
    num_augs = len(dataset.augs)
    base_labels = dataset.labels_raw
    weights = []
    for img_idx in range(len(base_labels)):
        race = base_labels.iloc[img_idx]
        w = float(race_weights.get(race, 1.0))
        weights.extend([w] * num_augs)

    weights = torch.as_tensor(weights, dtype=torch.double)
    assert len(weights) == len(dataset)

    g = torch.Generator()
    g.manual_seed(seed)
    sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True, generator=g)
    return sampler

race_weights = {
    "East Asian": 0.2,
    "Indian": 1.0,
    "Black": 1.0,
    "White": 1.0,
    "Middle Eastern": 1.0,
    "Latino_Hispanic": 1.0,
    "Southeast Asian": 1.0,
}

if __name__ == "__main__":
    print("Using device: ", device)
    if(torch.cuda.is_available()):
        print(f"GPU ID: {torch.cuda.current_device()}, {torch.cuda.get_device_name(torch.cuda.current_device())}")
    model = UResNet().to(device)
    
    if TRAINING:
        torch.autograd.set_detect_anomaly(True)
        with open("train_log.txt", "a") as file:
            file.write(f"Training on GPU ID: {torch.cuda.current_device()}, {torch.cuda.get_device_name(torch.cuda.current_device())}")

        train_dataset = FairFaceDataset(train_image_path, train_label_path)
        val_dataset = FairFaceDataset(val_image_path, val_label_path)
        train_loader = None #DataLoader(train_dataset, batch_size=B, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=B, shuffle=False, num_workers=8, pin_memory=True)

        for minority in ["All", "East Asian","Indian","Black","White","Middle Eastern","Latino_Hispanic","Southeast Asian"]:
            rm = None
            if minority == "All":
                rm = {r: 1.0 for r in race_weights.keys()}
            else:
                rm = {r: (0.2 if r == minority else 1.0) for r in race_weights.keys()}
            sampler = race_weighted_sampler(train_dataset, rm, num_samples=len(train_dataset), seed=42)
            train_loader = DataLoader(train_dataset, batch_size=B, shuffle=False, sampler=sampler, num_workers=8, pin_memory=True)

            # print("Number of training samples: ", len(train_dataset))
            # print("Number of validation samples: ", len(val_dataset))
            print(f"\n\n=== Training with minority: {minority} ===")
            with open("train_log.txt", "a") as file:
                file.write(f"\n\n=== Training with minority: {minority} ===\n")
                # file.write(f"\nNumber of training samples: {len(train_dataset)}")
                # file.write(f"\nNumber of validation samples: {len(val_dataset)}\n")
            
            print("Bootstrapping data loaders")
            for images, src, labels, label_str in train_loader:
                print("Batch of testing images shape: ", images.shape)
                print("Batch of source images shape: ", src.shape)
                print("Batch of labels shape: ", labels.shape)
                print("Batch of label strings: ", len(label_str))
                break

            train(
                model,
                train_loader,
                val_loader,
                stages=(["enc4"], ["enc4","enc3"], ["enc4","enc3","enc2"], ["entry","enc1","enc2","enc3","enc4"]),
                epochs_per_stage=(2, 2, 3, 1), #(1, 1, 1, 2),   # start small for testing
                lr=3e-5,
                out_dir="checkpoints_unet/minority_" + minority.replace(" ","_")
                # use_amp=True
            )

            del model
            torch.cuda.empty_cache()
    else:
        print("Load model from checkpoint for inference/testing")
        ckpt_path = "checkpoints_unet/best_stage4_epoch6.pt"
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"Loaded model from {ckpt_path} | val SSIM: {ckpt['val_ssim']:.4f}")
        model.to(device)

        #load test sample image
        img_file = ".//test_files//test_112.png"
        image = decode_image(img_file, mode = "RGB")
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((112, 112)),
            torchvision.transforms.ConvertImageDtype(torch.float),
            torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        input_img = transform(image).unsqueeze(0).to(device)  # add batch dimension
        with torch.no_grad():
            pred = model(input_img)
            # save output image
        output_img = denormalize_imagenet(pred.squeeze(0).cpu()).permute(1, 2, 0).numpy()
        output_pil = Image.fromarray(output_img)
        output_pil.save("test_output_unet.png")

