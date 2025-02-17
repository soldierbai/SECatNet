import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, data_dir, labels_dir, transform=None, device='cpu'):
        self.data_dir = data_dir
        labels_df = pd.read_excel(labels_dir, header=None)
        self.labels = labels_df.iloc[:, 0].values
        self.transform = transform
        self.images = []
        self.device = device

        for idx in range(len(self.labels)):
            img_name = os.path.join(self.data_dir, f"g{idx}.png")
            image = Image.open(img_name).convert('L')
            if self.transform:
                image = self.transform(image)
            self.images.append(image)

        self.images = [img.to(self.device) for img in self.images]
        self.labels = torch.tensor(self.labels, dtype=torch.long).to(self.device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label



