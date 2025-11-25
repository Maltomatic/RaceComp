from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from loaders.load_rfw import RFWDataset, train_label_path, val_image_path, val_label_path
from models.cnn_UNet_FR import ResUnetFR as UResNet

train_list = ["African", "Asian", "Caucasian", "Indian"]

B = 48

device = "cuda" if torch.cuda.is_available() else "cpu"

def rfw_accuracy(path_to_model):
    path_to_model = Path(path_to_model)

    df = pd.read_csv(train_label_path)

    for minority in train_list:
        val_dataset = RFWDataset(val_image_path, val_label_path, test_minority=minority, testing = True)
        val_loader = DataLoader(val_dataset, batch_size=B, num_workers=8, pin_memory=True)

        ckpt_file = tuple((path_to_model / f"minority_{minority}").iterdir())[0]
        race_ckpt = torch.load(ckpt_file, map_location=device)

        nm = race_ckpt["model"]["classifier.2.bias"].shape[0]

        model = UResNet(input_shape = (3, 224, 224), num_classes = nm).to(device)
        
        model.load_state_dict(race_ckpt["model"], strict = False)

        model.eval()

        with torch.no_grad():
            correct = 0
            
            for X_img, _, Y_label, *_ in val_loader:
                Y_label = Y_label.to(device).long()
                X_img = X_img.to(device).float()

                pred = model(X_img).argmax(dim=1)
                
                correct += (pred == Y_label).sum().item()

            accuracy = correct / len(val_dataset)

            print(f"RFW Accuracy for minority {minority}: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    rfw_accuracy("checkpoints_FR/UResNet/config_batchsize48_mb4")