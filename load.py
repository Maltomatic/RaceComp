import pandas as pd
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image

classes = ['East Asian', 'Indian', 'Black', 'White', 'Middle Eastern', 'Latino_Hispanic', 'Southeast Asian']
class_count = 7

train_image_path = val_image_path = "./dataset/fairface-img-margin025-trainval/"
train_label_path = "./dataset/fairface_label_train.csv"
val_label_path = "./dataset/fairface_label_val.csv"

class FairFaceDataset(Dataset):
    def __init__(self, image_path, label_path, transform=None):
        self.image_path = image_path
        self.file_path = pd.read_csv(label_path)["file"]
        self.labels_raw = pd.read_csv(label_path)["race"]

        cat = pd.Categorical(self.labels_raw, categories = classes, ordered = True)
        cat = pd.Series(cat.codes)
        # One-hot encode the labels
        self.labels = torch.nn.functional.one_hot(torch.tensor(cat).long(), num_classes=class_count)
        # print(self.labels.shape)
        # print(self.labels)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_file = os.path.join(self.image_path, self.file_path.iloc[idx])
        image = decode_image(img_file, mode = "RGB")
        # print(image.shape)
        # print(image)
        label_str = self.labels_raw.iloc[idx]
        label = self.labels[idx]
        downsample = None
        if self.transform:
            downsample = self.transform(image)

        return downsample, image, label, label_str
    
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((112,112)),
    torchvision.transforms.ConvertImageDtype(torch.float),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #imagenet parameters
])
train_dataset = FairFaceDataset(train_image_path, train_label_path, transform=transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = FairFaceDataset(val_image_path, val_label_path, transform=transforms)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

for images, src, labels, label_str in train_loader:
    print("Batch of testing images shape: ", images.shape)
    print("Batch of source images shape: ", src.shape)
    print("Batch of labels shape: ", labels.shape)
    print("Batch of label strings: ", len(label_str))
    break