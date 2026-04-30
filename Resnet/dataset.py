import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGE_SIZE = 256

JSON_BASE = "/home/fym/graduation/SAM-Med2D/data_cervical"
SPLIT_MAP = {
    20: "s1_sup20",
    50: "s1_sup50",
    100: "s1_sup100",
}


def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


class CervicalDataset(Dataset):
    def __init__(self, ratio, split="train", is_train=True):
        """
        ratio: 20, 50, or 100
        split: "train" or "test"
        """
        subdir = SPLIT_MAP[ratio]
        if split == "train":
            json_path = os.path.join(JSON_BASE, subdir, "image2label_train.json")
            with open(json_path) as f:
                data = json.load(f)
            self.image_paths = list(data.keys())
            self.mask_paths = [v[0] for v in data.values()]
        else:
            json_path = os.path.join(JSON_BASE, subdir, "label2image_test.json")
            with open(json_path) as f:
                data = json.load(f)
            self.mask_paths = list(data.keys())
            self.image_paths = list(data.values())

        self.transforms = get_transforms(is_train)
        print(f"[{split}] Loaded {len(self.image_paths)} slices (ratio={ratio}%)")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)

        augmented = self.transforms(image=img, mask=mask)
        img_t = augmented["image"]
        mask_t = augmented["mask"].unsqueeze(0).float()
        return img_t, mask_t
