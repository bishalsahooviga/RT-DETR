"""
RGBNDataset — 4-channel (RGB + NIR) COCO-format dataset for RT-DETR.

Images can be stored as:
  - NumPy .npy files: shape (H, W, 4) or (4, H, W), any dtype
  - GeoTIFF .tif files: read via rasterio if available, else tifffile
"""

import os
import numpy as np
import torch
import torch.utils.data

from pycocotools.coco import COCO
from torchvision import tv_tensors as datapoints

from src.core import register

__all__ = ['RGBNDataset']


def _load_image(path: str) -> np.ndarray:
    """Load a 4-channel RGBN image and return float32 array shaped (4, H, W) in [0, 1]."""
    ext = os.path.splitext(path)[1].lower()

    if ext == '.npy':
        img = np.load(path).astype(np.float32)
    elif ext in ('.tif', '.tiff'):
        try:
            import rasterio
            with rasterio.open(path) as src:
                img = src.read().astype(np.float32)   # (bands, H, W)
        except ImportError:
            import tifffile
            img = tifffile.imread(path).astype(np.float32)
            if img.ndim == 3 and img.shape[2] <= 8:  # (H, W, C) → (C, H, W)
                img = img.transpose(2, 0, 1)
    else:
        raise ValueError(f"Unsupported image format: {ext}. Expected .npy / .tif")

    # Ensure (C, H, W)
    if img.ndim == 3 and img.shape[0] not in (1, 2, 3, 4):
        img = img.transpose(2, 0, 1)

    # Normalise to [0, 1]
    if img.max() > 1.0:
        img = img / (255.0 if img.max() <= 255 else 65535.0)

    return img   # (4, H, W) float32 in [0, 1]


@register
class RGBNDataset(torch.utils.data.Dataset):
    """
    COCO-format dataset for 4-channel (RGB + NIR) imagery.

    Each image file must be either a .npy or .tif containing exactly 4 channels.
    Annotations follow the standard COCO JSON format.

    Args:
        img_folder (str):  Directory containing image files.
        ann_file (str):    Path to COCO-format JSON annotation file.
        transforms:        Injected torchvision v2 transform pipeline (Compose).
        return_masks (bool): Whether to load segmentation masks (default False).
    """

    __inject__ = ['transforms']
    __share__ = ['remap_mscoco_category']

    def __init__(self, img_folder, ann_file, transforms=None,
                 return_masks=False, remap_mscoco_category=False):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._transforms = transforms
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.ids)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        img_id = self.ids[idx]

        # ── Load image ──────────────────────────────────────────────
        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_folder, img_info['file_name'])
        img = _load_image(path)                    # (4, H, W) float32

        _, H, W = img.shape
        img_tensor = torch.from_numpy(img)         # (4, H, W)

        # ── Load annotations ────────────────────────────────────────
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        anns = [a for a in anns if a.get('iscrowd', 0) == 0]

        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = x, y, x + w, y + h
            # Clamp to image bounds
            x1, x2 = max(0, x1), min(W, x2)
            y1, y2 = max(0, y1), min(H, y2)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(ann['category_id'])

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)

        labels_t = torch.tensor(labels, dtype=torch.int64)

        # Wrap boxes as tv_tensors so v2 transforms handle them correctly
        boxes_tv = datapoints.BoundingBoxes(
            boxes_t,
            format=datapoints.BoundingBoxFormat.XYXY,
            spatial_size=(H, W),
        )

        target = {
            'boxes':     boxes_tv,
            'labels':    labels_t,
            'image_id':  torch.tensor([img_id]),
            'orig_size': torch.tensor([W, H]),
            'size':      torch.tensor([W, H]),
            'area':      torch.tensor([ann['area'] for ann in anns
                                       if ann.get('iscrowd', 0) == 0 and
                                          (ann['bbox'][2] > 0 and ann['bbox'][3] > 0)],
                                      dtype=torch.float32),
            'iscrowd':   torch.zeros(len(labels_t), dtype=torch.int64),
        }

        # ── Apply transforms ────────────────────────────────────────
        if self._transforms is not None:
            img_tensor, target = self._transforms(img_tensor, target)

        return img_tensor, target

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n num_images: {len(self.ids)}'
        return s