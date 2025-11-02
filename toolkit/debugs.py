import pandas as pd
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

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


def save_checkpoint(epoch, global_step, optimizer_step, model, optimizer, scaler, scheduler, accumulation_steps, ckpt_path, device):
    state = {
        "epoch": epoch,
        "global_step": global_step,       # micro-batch counter across epochs
        "optimizer_step": optimizer_step, # number of optimizer updates done
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "accumulation_steps": accumulation_steps
    }
    torch.save(state, ckpt_path)

def load_checkpoint(model, optimizer, scaler, scheduler, accumulation_steps, ckpt_path, device):
    try:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        scheduler.load_state_dict(state["scheduler"])
        return state["epoch"], state["global_step"], state["optimizer_step"], state["accumulation_steps"]
    except FileNotFoundError:
        return 0, 0, 0, accumulation_steps