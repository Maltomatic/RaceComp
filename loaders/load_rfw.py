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

train_image_path = "./dataset/RFW/train/"
val_image_path = "./dataset/RFW/test/"
train_label_path = "./dataset/rfw_train_labels.csv"
val_label_path = "./dataset/rfw_test_labels.csv"

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
        self.persons = df["person"]
        self.race_raw = df["race"]

        cat_race = pd.Categorical(self.race_raw, categories = classes, ordered = True)
        idx_race = torch.tensor(pd.Series(cat_race.codes).values).long()
        # One-hot encode the labels
        self.labels_race = torch.nn.functional.one_hot(idx_race, num_classes=class_count)
        #print(self.labels.shape)
        #print(self.labels)

        cat_person = pd.Categorical(self.persons)
        idx_person = torch.tensor(pd.Series(cat_person.codes).values).long()
        self.labels_person = torch.nn.functional.one_hot(idx_person, num_classes=len(cat_person.categories))

        norm = None
        if(normalize):
            norm = torchvision.transforms.Compose([
                    torchvision.transforms.ConvertImageDtype(torch.float),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #imagenet parameters
                ])
        else:
            norm = torchvision.transforms.ConvertImageDtype(torch.float)
        if(transform is None):
            transform = torchvision.transforms.Resize((224,224))
        self.transform = transform
        self.norm = norm
        self.augs = [
            None,
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomRotation(degrees=30),
            torchvision.transforms.RandomPerspective(),
            torchvision.transforms.RandomAffine(degrees=0, translate=(0.2,0.2))
        ]
        #verify size of each race
        print("Class distribution in RFW dataset:")
        print(self.race_raw.value_counts())

    def __len__(self):
        return len(self.labels_race)*len(self.augs)

    def __getitem__(self, idx):
        img_idx = idx // len(self.augs)
        aug_idx = idx % len(self.augs)

        img_file = os.path.join(self.image_path, self.file_path.iloc[img_idx])
        image = decode_image(img_file, mode = "RGB")
        # print(image.shape)
        # print(image)

        augmentation = "None"
        if aug_idx != 0:
            augmentation = self.augs[aug_idx]
            image = augmentation(image)
            # print(f"Applied augmentation: {augmentation}")

        race_str = self.race_raw.iloc[img_idx]
        race = self.labels_race[img_idx]
        person = self.labels_person[img_idx]
        image = self.norm(image)
        image = self.transform(image)

        return image, person, race, race_str

if __name__ == "__main__":
    train_dataset = RFWDataset(train_image_path, train_label_path)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

    test_dataset = RFWDataset(val_image_path, val_label_path)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=2)

    #verify all testing classes are in training set
    test_classes = set(test_dataset.persons)
    train_classes = set(train_dataset.persons)
    print("Number of classes in training set:", len(train_classes))
    print("Number of classes in testing set:", len(test_classes))
    missing_classes = test_classes - train_classes
    if len(missing_classes) == 0:
        print("All testing classes are present in the training set.")
    else:
        print("Missing classes in training set:", missing_classes)
        print("Number of missing classes:", len(missing_classes))

    # #test some images
    # for sample in [9876, 9877, 9878]:
    #     image, person, race, race_str = train_dataset[sample]
    #     print("Sampled image shape: ", image.shape)
    #     print("Sampled person ID one-hot: ", person)
    #     print("Sampled race one-hot: ", race)
    #     print("Sampled race string: ", race_str)

    for image, person, race, race_str in train_dataloader:
        print("Images shape: ", image.shape)
        print("Batch of person IDs shape: ", person.shape)
        print("Batch of races shape: ", race.shape)
        print("Length for batch of race strings: ", len(race_str))
        break
    