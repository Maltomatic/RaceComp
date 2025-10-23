import pandas as pd
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image

train_image_path = val_image_path = "./dataset/fairface-img-margin025-trainval/"
train_label_path = "./dataset/fairface_label_train.csv"
val_label_path = "./dataset/fairface_label_val.csv"

class FairFaceDataset(Dataset):
    def __init__(self, image_path, label_path, transform=None):
        self.image_path = image_path
        self.file_path = pd.read_csv(label_path)["file"]
        self.labels = pd.read_csv(label_path)[["race"]]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_file = os.path.join(self.image_path, self.file_path.iloc[idx])
        image = decode_image(img_file, mode = "RGB")
        # print(image.shape)
        # print(image)
        label = self.labels.iloc[idx]
        if self.transform:
            image = self.transform(image)

        return image, label
    
train_dataset = FairFaceDataset(train_image_path, train_label_path)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = FairFaceDataset(val_image_path, val_label_path)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)