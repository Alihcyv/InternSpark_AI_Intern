%%writefile dataset.py 

import os
import torch
import torchvision.transforms.v2 as v2

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


class ImageDataset(Dataset):
    def __init__(self, cfg):
        self.root_dir = cfg.ROOT_DIR
        self.classes = sorted(os.listdir(cfg.ROOT_DIR))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.images = []
        for cls_name in self.classes:
            cls_folder = os.path.join(cfg.ROOT_DIR, cls_name)
            if os.path.isdir(cls_folder):
                for img_name in os.listdir(cls_folder):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        path = os.path.join(cls_folder, img_name)
                        idx = self.class_to_idx[cls_name]
                        self.images.append((path, idx))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        return image, label


class TransformSubset(Dataset):
    def __init__(self, full_dataset, subset, transform=None):
        self.subset = subset
        self.full_dataset = full_dataset
        self.transform = transform

    def __getitem__(self, idx):
        actual_idx = self.subset[idx]
        image, label = self.full_dataset[actual_idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.subset)


def get_transform(cfg, is_train=True):
    norm = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if is_train:
        return v2.Compose([
        v2.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(10),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        norm,
    ])

    return v2.Compose([
        v2.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        norm
    ])


def prepare_dataloaders(full_dataset, cfg):
    labels = [item[1] for item in full_dataset.images]
    indices = list(range(len(full_dataset)))
    
    train_idx, temp_idx = train_test_split(
        indices, train_size = 0.8, stratify=labels, random_state=42
    )
    
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    train_dataset = TransformSubset(full_dataset, train_idx, transform=get_transform(cfg, is_train=True))
    val_dataset = TransformSubset(full_dataset, val_idx, transform=get_transform(cfg, is_train=False))
    test_dataset = TransformSubset(full_dataset, test_idx, transform=get_transform(cfg, is_train=False))
    
    print(f"All Images: {len(full_dataset)}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    kwargs = {'num_workers': 4, 'pin_memory': True}
        
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader
