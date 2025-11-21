import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image
from torch.utils.data import WeightedRandomSampler

classes = ['African', 'Asian', 'Caucasian', 'Indian']
class_count = 4

train_image_path = "./dataset/train/"
val_image_path = "./dataset/val/"
train_label_path = "./dataset/train_labels.csv"
val_label_path = "./dataset/val_labels.csv"

class RFWDataset(Dataset):
    def __init__(self, 
                 image_path, 
                 label_path, 
                 transform=None, 
                 normalize=True,
                 hr_size=(400, 400)):
        self.image_path = image_path
        df = pd.read_csv(label_path)
        self.file_path = df["file"]
        self.labels_raw = df["race"]


        cat = pd.Categorical(self.labels_raw, categories = classes, ordered = True)
        idx = torch.tensor(pd.Series(cat.codes).values).long()
        # One-hot encode the labels
        self.labels = torch.nn.functional.one_hot(idx, num_classes=class_count)
        #print(self.labels.shape)
        #print(self.labels)

        norm = None
        if(normalize):
            norm = torchvision.transforms.Compose([
                    torchvision.transforms.ConvertImageDtype(torch.float),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #imagenet parameters
                ])
        else:
            norm = torchvision.transforms.ConvertImageDtype(torch.float)
        if(transform is None):
            transform = torchvision.transforms.Resize((400,400))
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
        if(downsample.shape[1] != 400 or downsample.shape[2] != 400):
            downsample = torchvision.transforms.Resize((400,400))(downsample)
        # if not float tensor:
        if downsample.dtype != torch.float:
            downsample = torchvision.transforms.ConvertImageDtype(torch.float)(downsample)
        # print("Debug: After transform:", downsample.shape, downsample.dtype)

        return downsample, image, label, label_str

if __name__ == "__main__":
    dataset = RFWDataset(train_image_path, train_label_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    for downsample, src, labels, label_str in dataloader:
        print("Batch of testing images shape: ", downsample.shape)
        print("Batch of source images shape: ", src.shape)
        print("Batch of labels shape: ", labels.shape)
        print("Length for batch of label strings: ", len(label_str))
        break