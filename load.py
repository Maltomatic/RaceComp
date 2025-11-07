import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image
from torch.utils.data import WeightedRandomSampler

classes = ['East Asian', 'Indian', 'Black', 'White', 'Middle Eastern', 'Latino_Hispanic', 'Southeast Asian']
class_count = 7

train_image_path = val_image_path = "./dataset/fairface-img-margin025-trainval/"
train_label_path = "./dataset/fairface_label_train.csv"
val_label_path = "./dataset/fairface_label_val.csv"

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

class FairFaceDataset(Dataset):
    def __init__(self, 
                 image_path, 
                 label_path, 
                 transform=None, 
                 normalize=True,
                 lr_size=(112, 112),
                 hr_size=(224, 224)):
        self.image_path = image_path
        df = pd.read_csv(label_path)
        self.file_path = df["file"]
        self.labels_raw = df["race"]
        self.downsize = lr_size
        print(f"Dataset low-res size: {self.downsize}, high-res size: {hr_size}")

        cat = pd.Categorical(self.labels_raw, categories = classes, ordered = True)
        idx = torch.tensor(pd.Series(cat.codes).values).long()
        # One-hot encode the labels
        self.labels = torch.nn.functional.one_hot(idx, num_classes=class_count)
        # print(self.labels.shape)
        # print(self.labels)

        norm = None
        if(normalize):
            norm = torchvision.transforms.Compose([
                    torchvision.transforms.ConvertImageDtype(torch.float),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #imagenet parameters
                ])
        else:
            norm = torchvision.transforms.ConvertImageDtype(torch.float)
        if(transform is None):
            transform = torchvision.transforms.Resize(self.downsize)
        self.transform = transform
        self.norm = norm
        self.augs = [
            None,
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomRotation(degrees=179),
            torchvision.transforms.RandomPerspective(),
            torchvision.transforms.RandomAffine(degrees=0, translate=(0.2,0.2))
        ]

    def __len__(self):
        return len(self.labels)*len(self.augs)

    def __getitem__(self, idx):
        img_idx = idx // len(self.augs)
        aug_idx = idx % len(self.augs)

        img_file = os.path.join(self.image_path, self.file_path.iloc[img_idx])
        image = decode_image(img_file, mode = "RGB")
        # print(image.shape)
        # print(image)

        if aug_idx != 0:
            augmentation = self.augs[aug_idx]
            image = augmentation(image)
            # print(f"Applied augmentation: {augmentation}")

        label_str = self.labels_raw.iloc[img_idx]
        label = self.labels[img_idx]
        downsample = self.transform(image)
        downsample = self.norm(downsample)
        image = self.norm(image)
        # print("Debug: After norm:", downsample.shape, downsample.dtype)
        # print("Debug: Before transform:", image.shape, image.dtype)
        if(downsample.shape[1] != self.downsize[0] or downsample.shape[2] != self.downsize[1]):
            downsample = torchvision.transforms.Resize(self.downsize)(downsample)
        # if not float tensor:
        if downsample.dtype != torch.float:
            downsample = torchvision.transforms.ConvertImageDtype(torch.float)(downsample)
        # print("Debug: After transform:", downsample.shape, downsample.dtype)

        return downsample, image, label, label_str

if __name__ == "__main__":
    dataset = FairFaceDataset(train_image_path, train_label_path, lr_size = (56, 56))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    for downsample, src, labels, label_str in dataloader:
        print("Batch of testing images shape: ", downsample.shape)
        print("Batch of source images shape: ", src.shape)
        print("Batch of labels shape: ", labels.shape)
        print("Length for batch of label strings: ", len(label_str))
        break