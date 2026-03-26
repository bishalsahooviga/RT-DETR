# src/data/dataset/rgbn_ship_dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os

class RGBNShipDataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_dir, img_info['file_name'])

        # Load 4-channel RGBN array: shape (H, W, 4) or (4, H, W)
        img = np.load(path).astype(np.float32)

        # Normalize to [0, 1] if raw uint16/uint8
        if img.max() > 1.0:
            img = img / 255.0 if img.max() <= 255 else img / 65535.0

        # Ensure shape is (H, W, 4)
        if img.shape[0] == 4:
            img = img.transpose(1, 2, 0)

        # Build target dict (COCO format)
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])  # xyxy
            labels.append(ann['category_id'])

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'image_id': torch.tensor([img_id]),
        }

        # Convert to tensor (C, H, W)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1))

        if self.transforms:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target

    def __len__(self):
        return len(self.ids)