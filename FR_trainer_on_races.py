import pandas as pd
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import warnings

import torch, torchvision
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image
import math
import time
import copy
from datetime import datetime
from collections import defaultdict
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from loaders.load_rfw import RFWDataset
from loaders.load_rfw import races, race_count, train_image_path, train_label_path, val_image_path, val_label_path

from models.cnn_UNet_FR import ResUnetFR as UResNet
from models.vit_FR import ViTFR as VitNet

B = 32
C = 3
H = W = 224
# race_count = 4  # race classification: 4 classes (African, Asian, Caucasian, Indian)  <<< COMMENT

#################### configs #################### 
TRAINING = True
debug = False
resume = False
custom_load = False
weight_path = "checkpoints/unet_FR_base.pt"
training_comment = "FR ViT training on RFW (Lawrance Colab run)"

model_idx = 0
# idx:
    # 0 - VitNet
    # 1 - UResNet
train_list = ["African", "Asian", "Caucasian", "Indian"]
microbatches = 4 # 1 for no microbatching, n for n-step microbatching, max 8 recommended to avoid gradient explosion
sz = 224 # or 56
epoch_stages = (3, 2, 2, 0)

under_represented_ratio = 0.05
tgt_race = "All"
test_stage = 2
test_epoch = 3

config_str = f"batchsize{B}_mb{microbatches}"
#################################################
model_names = ["VitNet", "UResNet"]

model_type = model_names[model_idx]
desc_path = f"FR_{model_type}/{config_str}/"
desc = f"trained_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

train_stages = (["enc4"], ["enc4","enc3"], ["enc4","enc3","enc2"], ["entry","enc1","enc2","enc3","enc4"]) if sz == 112\
    else (["enc3"], ["enc3","enc2"], ["enc1","enc2","enc3"])

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = "cuda" if torch.cuda.is_available() else "cpu"

torch.autograd.set_detect_anomaly(True)

def imagenet_denorm(x):
    mean = x.new_tensor(IMAGENET_MEAN).view(1,-1,1,1)
    std  = x.new_tensor(IMAGENET_STD).view(1,-1,1,1)
    return x * std + mean

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

def accumulate_by_race(bucket, race, loss, correct, total):
    b = bucket.setdefault(race, {"loss": [], "correct": 0, "total": 0})
    b["loss"].append(loss)
    b["correct"] += correct
    b["total"] += total
    

def train(model, 
          train_loader, 
          val_loader, 
          stages=(["enc4"], ["enc4","enc3"], ["enc4","enc3","enc2"], ["entry","enc1","enc2","enc3","enc4"]),
          epochs_per_stage=(2, 2, 2, 4),
          lr=0.003,
          out_dir="checkpoints",
          microbatch_steps = 8,
          resume = False,
          AMP_en = False):

    os.makedirs(out_dir, exist_ok=True)
    model = model.to(device)
    scaler = torch.amp.GradScaler(device_type, enabled=AMP_en)

    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    global_epoch = 0
    start_epoch = 0
    resume_batch = 0

    if(resume):
        svpt = torch.load('savepoints_FR/ckpt_s0_e1_b13536.pt', weights_only=False)

        print("Verifying checkpoint keys: ", svpt.keys())
        model.load_state_dict(svpt['model_state'])
        print("Resumed model state from savepoint.")
        print(f"Please ensure testing minority is {svpt['race']}")
    try:
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
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs * math.ceil(len(train_loader) / microbatch_steps)))
            
            if(resume and stage_idx == svpt["stage"]):
                start_epoch = svpt['epoch'] - 1
                global_epoch += start_epoch
                resume_batch = svpt['batch']
                tr = svpt['losses']
                print(f"Resumed optimizer and scheduler state from savepoint at stage {svpt['stage']}, epoch {global_epoch}, batch {resume_batch}.")
            elif (resume and stage_idx < svpt["stage"]):
                print(f"Skipping stage {stage_idx}, resume savepoint at stage {svpt['stage']}.")
                global_epoch += total_epochs
                continue  # skip earlier stages if resuming

            print(f"\n=== Stage {stage_idx+1}/{len(stages)} | Unfrozen: {layer_list} ===")
            with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
                file.write(f"\n=== Stage {stage_idx+1}/{len(stages)} | Unfrozen: {layer_list} ===\n")

            for e in range(start_epoch, total_epochs):
                global_epoch += 1
                print(f"\n--- Epoch {e+1}/{total_epochs} time: {datetime.now().strftime('%H:%M:%S')}---")
                
                with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
                    file.write(f"\n--- Epoch {global_epoch} || time: {datetime.now().strftime('%H:%M:%S')} --- \n")
                model.train()
                if(not resume):
                    tr = defaultdict(float)
                    tr["correct"] = 0.0
                    tr["total"] = 0.0
                optimizer.zero_grad(set_to_none=True)
                n_batches = 0

                if(debug):
                    print("Debug: Skipping training loop in debug mode.")
                    continue

                # unpack RFWDataset as (image, person_idx, person_raw, race_onehot, race_str)
                for X_img, person_label, person_raw, race_tensor, race_str in train_loader:
                    race_tensor = race_tensor.to(device)
                    Y_label = race_tensor.argmax(dim=1).long() # convert race one-hot -> class indices (0..3)
                    if(n_batches < resume_batch and resume):
                        n_batches += 1
                        continue
                    elif(resume):
                        resume = False
                        scheduler.load_state_dict(svpt['scheduler_state'])
                        # optimizer.load_state_dict(svpt['optim_state'])
                        # scaler.load_state_dict(svpt['scaler_state'])
                        print(f"Skipped to batch {n_batches} ({resume_batch}) to resume from savepoint.")
                        #verify
                        print("==================================================================")
                        print(f"Currently at stage {stage_idx}, epoch {e}, batch {n_batches}.")
                        print(f"Start epoch {start_epoch}, global epoch {global_epoch}, total epoch {total_epochs}.")

                    
                    X_img = torch.nan_to_num(X_img, nan=0.0, posinf=1e10, neginf=-1e10)
                    X_img = X_img.to(device).float()

                    with torch.amp.autocast(device_type, enabled=AMP_en):
                        # print("Debug: Input shapes:", X_img.shape, Y_label.shape)
                        pred = model(X_img)
                        if not torch.isfinite(pred).all():
                            print(f"----WARNING: [Batch {n_batches}] Returneed infinite logits; skipping")
                            optimizer.zero_grad(set_to_none=True)
                            n_batches -= (n_batches% microbatch_steps)  # reset microbatch count
                            continue
                        pred = torch.clamp(pred, min=-100, max=100)
                        pixel_loss = criterion(pred, Y_label)          # <<< still CE over race indices
                        loss = pixel_loss / microbatch_steps  # Scale loss for gradient accumulation
                        # for name, param in model.named_parameters():
                        #     if param.grad is not None and torch.isnan(param.grad).any():
                        #         print(f"NaN gradient in {name}")
                        if not torch.isfinite(loss).all():
                            print(f"----WARNING: [Batch {n_batches}] Returneed infinite loss; skipping")
                            optimizer.zero_grad(set_to_none=True)
                            n_batches -= (n_batches% microbatch_steps)  # reset microbatch count
                            continue
                    
                    with torch.no_grad():
                        preds = pred.argmax(dim=1)
                        correct = (preds == Y_label).sum().item()
                        tr["correct"] += correct
                        tr["total"] += Y_label.size(0)

                    scaler.scale(loss).backward()

                    if((n_batches + 1) % microbatch_steps == 0 or (n_batches + 1) == len(train_loader)):
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                        scheduler.step()
                    
                    tr["loss"] += loss.item() * microbatch_steps # unscale the loss

                    n_batches += 1
                    # if(n_batches % 1 == 0):
                    if(n_batches % 50 == 1):
                        print(f"Batch {n_batches:03d} | train: loss {tr['loss']/n_batches:.4f}")
                        print(f"At step {n_batches + 1}, Learning rate {scheduler.get_last_lr()[0]}")
                    if(n_batches % 2000 == 1):
                        with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
                            file.write(f"Batch {n_batches:03d} at time {datetime.now().strftime('%H:%M:%S')} | train: loss {tr['loss']/n_batches:.4f}\n")
                train_loss = tr["loss"] / n_batches
                train_acc = tr["correct"] / tr["total"] if tr["total"] > 0 else 0.0
                print(f"Epoch {global_epoch:03d} | train: loss {train_loss:.4f} | acc {train_acc:.4f}")
                with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
                    file.write(f"Epoch {global_epoch:03d} | train: loss {train_loss:.4f} | acc {train_acc:.4f}\n")
                
                model.eval()
                print(f"==Validation: || time: {datetime.now().strftime('%H:%M:%S')} ==")
                with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
                    file.write(f"==Validation: || time: {datetime.now().strftime('%H:%M:%S')} ==\n")
                v = defaultdict(float); n_val = 0; race_bucket = {}
                v["correct"] = 0.0
                v["total"] = 0.0
                with torch.no_grad():
                    for X_img, person_label, person_raw, race_tensor, race_str in val_loader:
                        race_tensor = race_tensor.to(device)
                        Y_label = race_tensor.argmax(dim=1).long()

                        X_img = X_img.to(device).float()

                        pred = model(X_img)
                        val_loss = criterion(pred, Y_label)

                        v["loss"] += val_loss.item()
                        n_val += 1

                        preds = pred.argmax(dim=1)
                        correct_vec = (preds == Y_label)
                        v["correct"] += correct_vec.sum().item()
                        v["total"] += Y_label.size(0)

                        val_loss_vec = F.cross_entropy(pred, Y_label, reduction="none")
                        # race_str is the tuple of race names
                        for i, r in enumerate(race_str):
                            loss_i = val_loss_vec[i].item()
                            correct_i = 1 if correct_vec[i].item() else 0
                            accumulate_by_race(race_bucket, r, loss_i, correct_i, 1)


                val_loss = v["loss"]/n_val
                val_acc = v["correct"]/v["total"] if v["total"] > 0 else 0.0


                print(f"           val:   loss {val_loss:.4f} | acc {val_acc:.4f}")
                with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
                    file.write(f"val: loss {val_loss:.4f} | acc {val_acc:.4f}\n")   


                race_summary = {}
                for k, stats in race_bucket.items():
                    loss_list = stats["loss"]
                    total = stats["total"]
                    correct = stats["correct"]

                    race_loss = float(np.mean(loss_list)) if len(loss_list) > 0 else float("nan")
                    race_acc = float(correct / total) if total > 0 else float("nan")

                    race_summary[k] = {
                        "loss": race_loss,
                        "acc": race_acc,
                    }

                if race_summary:
                    print("           per-race (val):",
                        "  ".join([f"{k}: Loss {vals['loss']:.3f}, Acc {vals['acc']:.3f}"
                                    for k, vals in race_summary.items()]))

                    with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
                        file.write(
                            "per-race (val): "
                            + "  ".join([f"{k}: Loss {vals['loss']:.3f}, Acc {vals['acc']:.3f}"
                                        for k, vals in race_summary.items()])
                            + "\n"
                        )
            
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    ckpt = {
                        "model": model.state_dict(),
                        "val_loss": val_loss,
                        "race_summary": race_summary,
                        "stage": stage_idx+1, "epoch": global_epoch
                    }
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir, exist_ok=True)
                    path = os.path.join(out_dir, f"best_stage{stage_idx+1}_epoch{global_epoch}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt")
                    torch.save(ckpt, path)
                    print(f"Saved best checkpoint → {path} (Loss={val_loss:.4f})")
            del optimizer
            del scheduler
    except:
        print("Save checkpoint before quitting? (y/n)")
        if(input().lower().startswith('y')):
            if not os.path.exists('savepoints'):
                os.makedirs('savepoints', exist_ok=True)
            ckpt_epoch = global_epoch
            ckpt_stage = stage_idx
            ckpt_batch = n_batches
            ckpt_race = minority
            ckpt_model_state = model.state_dict()
            ckpt_optim_state = optimizer.state_dict()
            ckpt_scheduler_state = scheduler.state_dict()
            ckpt_scaler_state = scaler.state_dict()
            ckpt_tr = tr
            torch.save({
                "model_state": ckpt_model_state,
                "optim_state": ckpt_optim_state,
                "scheduler_state": ckpt_scheduler_state,
                "scaler_state": ckpt_scaler_state,
                "epoch": ckpt_epoch,
                "batch": ckpt_batch,
                "stage": ckpt_stage,
                "race": ckpt_race,
                "losses": ckpt_tr
            }, f'savepoints/ckpt_s{ckpt_stage}_e{ckpt_epoch}_b{ckpt_batch}.pt')
        raise


race_weights = {
    "East Asian": 1.0,
    "Indian": 1.0,
    "Black": 1.0,
    "White": 1.0,
    "Middle Eastern": 1.0,
    "Latino_Hispanic": 1.0,
    "Southeast Asian": 1.0,
}

if __name__ == "__main__":
    print("Using device: ", device)
    print(f"Logging to logs/training/{desc_path}{desc}.txt with configs:\n - Model: {model_type}, Batch size: {B}, Microbatch steps: {microbatches}")
    if(torch.cuda.is_available()):
        print(f"GPU ID: {torch.cuda.current_device()}, {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    torch.autograd.set_detect_anomaly(True)
    if not os.path.exists(f"./logs/training/{desc_path}"):
        os.makedirs(f"./logs/training/{desc_path}", exist_ok=True)
    with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
        file.write(f"\n\n=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} : {training_comment} ===\n")
        file.write(f"Training on GPU ID: {torch.cuda.current_device()}, {torch.cuda.get_device_name(torch.cuda.current_device())}")
        file.write(f"\nConfigs - Model: {model_type}, Batch size: {B}, Microbatch steps: {microbatches}")

    for minority in train_list:
        print(f"\n\nPreparing training dataset for minority: {minority}")
        train_dataset = RFWDataset(train_image_path, train_label_path, minority=minority) if minority != "All" else \
                        RFWDataset(train_image_path, train_label_path)
        train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True, num_workers=8, pin_memory=True)
        

        limit_classes = set(train_dataset.minority_classes) if minority != "All" else None

        Modelnet = VitNet(input_shape = (3, sz, sz), num_classes = race_count) if model_idx == 0 else \
                UResNet(input_shape = (3, 224, 224), num_classes = race_count) 

        print(f"Model initialized with {race_count} output classes.")
        with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
            file.write(f"\nModel initialized with {race_count} output classes.\n")
        
        print("Creating validation dataset and loader.")
        val_dataset = RFWDataset(val_image_path, val_label_path, restrict_classes=limit_classes, test_minority=minority, testing = True) if minority != "All" else \
                        RFWDataset(val_image_path, val_label_path, testing = True)
        val_loader = DataLoader(val_dataset, batch_size=B, shuffle=True, num_workers=8, pin_memory=True)
        # verify all val classes are in training set
        val_classes = set(val_dataset.persons)
        train_classes = set(train_dataset.persons)
        missing_classes = val_classes - train_classes
        if len(missing_classes) > 0:
            print(f"Warning: Missing classes in training set for minority {minority}: ", missing_classes)
            print("Skipping this minority.")
            continue

        model = copy.deepcopy(Modelnet)
        for par1, par2 in zip(model.parameters(), Modelnet.parameters()):
            assert torch.allclose(par1, par2), "Model copy failed!"
        #load racially biased weights with strict=False to allow partial loading
        if(custom_load):
            print("loading")
            race_ckpt = torch.load(weight_path, map_location=device)
            model.load_state_dict(race_ckpt["model"], strict = False)
            print("loaded")
        
        model.to(device)
        print(f"Created copy of {model_type} initialized for training.")

        print("Number of training samples: ", len(train_dataset))
        print("Number of validation samples: ", len(val_dataset))

        print(f"\n=== Training with minority: {minority} ===")
        with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
            file.write(f"\n=== Training with minority: {minority} ===\n")
        
        print("Bootstrapping data loaders")
        for images, label_idx, person_raw, race_tensor, race_str in train_loader:
            print("Batch of testing images shape: ", images.shape)
            print("Batch of int person labels shape: ", label_idx.shape)
            print("Example person strings: ", person_raw[:5])
            print("Batch of race tensor shape: ", race_tensor.shape)
            print("Example race strings: ", race_str[:5])
            break
                
        train(
            model,
            train_loader,
            val_loader,
            stages=train_stages,
            epochs_per_stage= epoch_stages, #(2, 2, 3, 1),
            lr=9e-5,
            out_dir=f"checkpoints_FR/{model_type}/config_{config_str}/minority_{minority.replace(' ','_')}",
            microbatch_steps = microbatches,
            resume = resume
        )
        resume = False

        print(f"Finished training for minority: {minority}, releasing model from GPU.\n\n")
        with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
            file.write(f"\nFinished training for minority: {minority}.\n\n")

        del model
        torch.cuda.empty_cache()
