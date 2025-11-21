import pandas as pd
import os
import torch, torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image

classes = ['East Asian', 'Indian', 'Black', 'White', 'Middle Eastern', 'Latino_Hispanic', 'Southeast Asian']
class_count = 7

train_image_path = val_image_path = "./dataset/fairface-img-margin025-trainval/"
train_label_path = "./dataset/fairface_label_train.csv"
val_label_path = "./dataset/fairface_label_val.csv"

class FairFaceDataset_Trim(Dataset):
    def __init__(self, 
                 image_path, 
                 label_path, 
                 transform=None, 
                 normalize=True,
                 lr_size=(112, 112),
                 hr_size=(224, 224),
                 minority='All',
                 minority_weight = 0.05):
        self.minority = minority
        self.minority_weight = minority_weight

        self.image_path = image_path
        df = pd.read_csv(label_path)
        print(f"Original dataset size: {len(df)}")
        print("Original class distribution:")
        print(df["race"].value_counts())
        if(minority != 'All'):
            min_df = df[df["race"] == minority]
            full_df = df[df["race"] != minority]
            df = pd.concat([full_df, min_df.sample(frac=min(1.0, minority_weight), replace=False)], ignore_index=True) 
        self.file_path = df["file"]
        self.labels_raw = df["race"]
        if(minority != 'All'):
            print(f"Dataset size after applying minority weight ({minority}: {minority_weight}): {len(self.labels_raw)}")
            print("Verify class distribution after applying minority weight:")
            print(self.labels_raw.value_counts())

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
    race_weights = {
        "East Asian": 0.2,
        "Indian": 1.0,
        "Black": 1.0,
        "White": 1.0,
        "Middle Eastern": 1.0,
        "Latino_Hispanic": 1.0,
        "Southeast Asian": 1.0,
    }
    minority = 'Middle Eastern'
    rm = {r: (0.05 if r == minority else 1.0) for r in race_weights.keys()}

    dataset = FairFaceDataset_Trim(train_image_path, train_label_path, lr_size = (56, 56), minority=minority, minority_weight=0.05)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    for downsample, src, labels, label_str in dataloader:
        print("Batch of testing images shape: ", downsample.shape)
        print("Batch of source images shape: ", src.shape)
        print("Batch of labels shape: ", labels.shape)
        print("Length for batch of label strings: ", len(label_str))
        break