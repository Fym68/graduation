import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class CervicalSliceDataset(Dataset):
    """Binary classification dataset: tumor (pos) vs no-tumor (neg) slices."""

    def __init__(self, data_dir, split="train", transform=None, patient_ids=None):
        """
        Args:
            data_dir: root dir containing train/test -> positive/negative
            split: "train" or "test"
            transform: torchvision transforms
            patient_ids: if provided, only include these patients (for train/val split)
        """
        self.transform = transform
        self.samples = []  # (path, label)

        pos_dir = os.path.join(data_dir, split, "positive")
        neg_dir = os.path.join(data_dir, split, "negative")

        for fname in sorted(os.listdir(pos_dir)):
            if not fname.endswith("_pos.png"):
                continue
            pid = fname.split("-T2S-")[0]
            if patient_ids is not None and pid not in patient_ids:
                continue
            self.samples.append((os.path.join(pos_dir, fname), 1))

        for fname in sorted(os.listdir(neg_dir)):
            if not fname.endswith("_neg.png"):
                continue
            pid = fname.split("-T2S-")[0]
            if patient_ids is not None and pid not in patient_ids:
                continue
            self.samples.append((os.path.join(neg_dir, fname), 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if self.transform:
            img = self.transform(img)
        return img, label, path


def get_patient_ids(data_dir, split="train"):
    """Extract unique patient IDs from a split."""
    pids = set()
    for subdir in ["positive", "negative"]:
        d = os.path.join(data_dir, split, subdir)
        for fname in os.listdir(d):
            if fname.endswith(("_pos.png", "_neg.png")):
                pids.add(fname.split("-T2S-")[0])
    return sorted(pids)


def get_transforms(is_train=True, image_size=224):
    if is_train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
