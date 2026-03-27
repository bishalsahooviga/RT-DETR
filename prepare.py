"""
RT-DETRv2 Data Preparation Pipeline
=====================================
Handles 4-band GeoTIFF (R, G, B, NIR) + GeoJSON polygon annotations.

Pipeline stages:
  1. Normalize & stack 4-channel RGBN → processed GeoTIFFs
  2. Reproject GeoJSON polygons → pixel-space bounding boxes
  3. Tile large images (1024×1024, 25% overlap) → .npy patches
  4. Augment each tile 8× with bbox-aware transforms
  5. Split by SOURCE IMAGE (train 80 / val 15 / test 5)
  6. Export COCO JSON per split

Directory layout produced:
  <ROOT>/
    processed_tifs/      ← normalized 4-band GeoTIFFs
    bboxes/              ← per-image pixel bbox JSONs
    tiles/               ← raw .npy tiles  +  tile bbox JSONs
    augmented/           ← 8 augmented copies per tile
    dataset/
      train/  val/  test/   ← final .npy files
      train.json  val.json  test.json  ← COCO annotations

Requirements:
  pip install rasterio geopandas shapely albumentations tqdm numpy
"""

# ─────────────────────────────────────────────
# 0. Imports & Config
# ─────────────────────────────────────────────
import json
import os
import random
import shutil
from pathlib import Path

import albumentations as A
import cv2
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.transform
from tqdm import tqdm

# ── Paths ──────────────────────────────────────
ROOT_DIR      = Path.home() / "bhunayan" / "Bhunayan2" / "DEV"
TIF_DIR       = ROOT_DIR / "tifs"
JSON_DIR      = ROOT_DIR / "jsons"
PROC_TIF_DIR  = ROOT_DIR / "processed_tifs"
BBOX_DIR      = ROOT_DIR / "bboxes"
TILE_DIR      = ROOT_DIR / "tiles"
AUG_DIR       = ROOT_DIR / "augmented"
DATASET_DIR   = ROOT_DIR / "dataset"

# ── Tiling ─────────────────────────────────────
TILE_SIZE     = 1024          # px
OVERLAP       = 0.25          # 25 % → 256 px stride overlap
MIN_VIS_RATIO = 0.30          # drop box if < 30 % visible in tile

# ── Augmentation ───────────────────────────────
AUG_COPIES    = 8             # augmented versions per tile

# ── Dataset split (by source image, not by tile) ──
TRAIN_RATIO   = 0.80
VAL_RATIO     = 0.15
# TEST_RATIO  = 1 - TRAIN - VAL  (implicit)

RANDOM_SEED   = 42

# ── Category ───────────────────────────────────
CATEGORIES = [
    {"id": 1, "name": "Boat"},
    {"id": 2, "name": "Vehicle-FM"},
    {"id": 3, "name": "Rail Car"},
    {"id": 4, "name": "Sheds-C"},
    {"id": 5, "name": "Sheds-T"},
    {"id": 6, "name": "Ship-L"},
    {"id": 7, "name": "Ship-S"},
    {"id": 8, "name": "Swimming Pool"},
    {"id": 9, "name": "Water Tank"},
    {"id": 10, "name": "Ground-C"},
    {"id": 11, "name": "Ground-O"},
    {"id": 12, "name": "STP-F"},
    {"id": 13, "name": "Bridge-1"},
    {"id": 14, "name": "Bridge-2"},
    {"id": 15, "name": "Bridge-3"},
    {"id": 16, "name": "Bus Station"},
    {"id": 17, "name": "Brick Kiln-C"},
    {"id": 18, "name": "Brick Kiln-R"},
    {"id": 19, "name": "Metroshed-1"},
    {"id": 20, "name": "Metroshed-2"},
    {"id": 21, "name": "Solar Panels"},
    {"id": 22, "name": "TOLL"},
    {"id": 23, "name": "Vehicles-B"},
    {"id": 24, "name": "Wind Turbine"}
]

# ── Category lookup (name → id) ────────────────
CATEGORY_NAME_TO_ID: dict[str, int] = {str(cat["name"]): int(cat["id"]) for cat in CATEGORIES}

# ─────────────────────────────────────────────
# Helper: extract category_id from a GeoDataFrame row
# ─────────────────────────────────────────────
_LABEL_KEYS = ("class", "Class", "label", "Label", "category",
               "Category", "name", "Name", "type", "Type")


def _get_category_id(row) -> int:
    """
    Try common property-key names to find the object class in a GeoJSON row.
    Returns the matching COCO category id (1-based), or 1 if none is found.
    """
    for key in _LABEL_KEYS:
        val = getattr(row, key, None)
        if val is None or (hasattr(val, 'is_valid')):
            continue                              # skip geometry columns
        val_str = str(val).strip()
        cat_id = CATEGORY_NAME_TO_ID.get(val_str)
        if cat_id is not None:
            return cat_id
    return 1                                     # default: first category


# ─────────────────────────────────────────────
# Helper: create all output directories up front
# ─────────────────────────────────────────────
def make_dirs():
    for d in [PROC_TIF_DIR, BBOX_DIR, TILE_DIR, AUG_DIR,
              DATASET_DIR / "train", DATASET_DIR / "val", DATASET_DIR / "test"]:
        d.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════
# STAGE 1 — Normalize & Save 4-Channel GeoTIFF
# ═════════════════════════════════════════════
def percentile_norm(band: np.ndarray) -> np.ndarray:
    """
    Stretch a single float32 band to uint8 using 2nd–98th percentile clipping.
    This is robust to extreme outliers common in remote-sensing imagery.
    """
    p2  = np.percentile(band, 2)
    p98 = np.percentile(band, 98)
    stretched = (band - p2) / (p98 - p2 + 1e-6) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)


def load_rgbn(tif_path: Path) -> np.ndarray:
    """
    Read bands 1-4 from a GeoTIFF, normalize each independently,
    and return an H×W×4 uint8 array in (R, G, B, NIR) order.

    Independent per-band normalization matters because NIR has a very
    different dynamic range than visible bands.
    """
    with rasterio.open(tif_path) as src:
        if src.count < 4:
            raise ValueError(
                f"{tif_path.name} has only {src.count} band(s); expected 4 (R,G,B,NIR)."
            )
        r   = src.read(1).astype(np.float32)
        g   = src.read(2).astype(np.float32)
        b   = src.read(3).astype(np.float32)
        nir = src.read(4).astype(np.float32)

    return np.stack([percentile_norm(r),
                     percentile_norm(g),
                     percentile_norm(b),
                     percentile_norm(nir)], axis=-1)   # H×W×4


def save_rgbn_geotiff(src_path: Path, dst_path: Path, img: np.ndarray):
    """
    Write a uint8 H×W×4 array as a 4-band GeoTIFF, preserving the
    original CRS and spatial transform so annotations stay aligned.

    img: H×W×4 uint8 (R, G, B, NIR)
    """
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()

    profile.update(dtype=rasterio.uint8, count=4)

    with rasterio.open(dst_path, "w", **profile) as dst:
        for band_idx in range(4):                      # bands are 1-indexed in rasterio
            dst.write(img[:, :, band_idx], band_idx + 1)


def stage1_normalize_tifs():
    """
    Iterate over every .tif in TIF_DIR, normalize to uint8, and save
    the result to PROC_TIF_DIR with the same filename.
    Skips files that have already been processed.
    """
    print("\n── Stage 1: Normalizing TIF files ──────────────────────────")
    tif_files = sorted(TIF_DIR.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {TIF_DIR}")

    for tif_path in tqdm(tif_files, desc="Normalizing TIFs"):
        out_path = PROC_TIF_DIR / tif_path.name
        if out_path.exists():
            continue                                   # skip if already done
        img = load_rgbn(tif_path)
        save_rgbn_geotiff(tif_path, out_path, img)

    print(f"  ✓ {len(tif_files)} TIF(s) processed → {PROC_TIF_DIR}")


# ═════════════════════════════════════════════
# STAGE 2 — GeoJSON Polygons → Pixel BBoxes
# ═════════════════════════════════════════════
def geo_to_pixel_bboxes(geojson_path: Path, tif_path: Path) -> list[dict]:
    """
    Convert GeoJSON polygon annotations to pixel-space bounding boxes.

    Two modes depending on whether the GeoJSON has a CRS:

    • No CRS (pixel-space annotations, as produced by many labelling tools):
      Coordinates are treated directly as (col, row) pixel values.
      The TIF's affine transform is NOT applied — coords are already pixels.

    • With CRS (true geographic annotations):
      Reproject the GeoDataFrame into the TIF's CRS, then use rasterio's
      rowcol() to map geographic → pixel coordinates.

    Returns a list of dicts: {x_min, y_min, x_max, y_max} in pixel space.
    """
    with rasterio.open(tif_path) as src:
        transform = src.transform
        crs       = src.crs
        img_h     = src.height
        img_w     = src.width

    gdf = gpd.read_file(geojson_path)

    bboxes = []

    if gdf.crs is None or crs is None:
        # ── Pixel-space annotations (no CRS on GeoJSON or TIF) ───────────
        # Coordinates are treated as (col, row) pixel values directly.
        for row in gdf.itertuples(index=False):
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            cat_id = _get_category_id(row)
            minx, miny, maxx, maxy = geom.bounds  # minx=col_min, miny=row_min

            x_min = int(np.clip(minx, 0, img_w - 1))
            y_min = int(np.clip(miny, 0, img_h - 1))
            x_max = int(np.clip(maxx, 0, img_w - 1))
            y_max = int(np.clip(maxy, 0, img_h - 1))

            if x_max <= x_min or y_max <= y_min:
                continue

            bboxes.append({"x_min": x_min, "y_min": y_min,
                            "x_max": x_max, "y_max": y_max,
                            "category_id": cat_id})

    else:
        # ── Geographic annotations: reproject then convert via transform ──
        gdf = gdf.to_crs(crs)

        for row in gdf.itertuples(index=False):
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            cat_id = _get_category_id(row)
            minx, miny, maxx, maxy = geom.bounds

            # top-left corner of bbox
            row_min, col_min = rasterio.transform.rowcol(transform, minx, maxy)
            # bottom-right corner of bbox
            row_max, col_max = rasterio.transform.rowcol(transform, maxx, miny)

            # Clamp to image bounds
            x_min = int(np.clip(col_min, 0, img_w - 1))
            y_min = int(np.clip(row_min, 0, img_h - 1))
            x_max = int(np.clip(col_max, 0, img_w - 1))
            y_max = int(np.clip(row_max, 0, img_h - 1))

            if x_max <= x_min or y_max <= y_min:
                continue

            bboxes.append({"x_min": x_min, "y_min": y_min,
                            "x_max": x_max, "y_max": y_max,
                            "category_id": cat_id})
    return bboxes


def stage2_extract_bboxes():
    """
    For each GeoJSON in JSON_DIR, find its matching processed TIF by
    base name and convert all polygon annotations to pixel bboxes.
    Results are saved as JSON files in BBOX_DIR.
    """
    print("\n── Stage 2: Converting GeoJSON polygons → pixel bboxes ─────")
    json_files = sorted(JSON_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {JSON_DIR}")

    missing_tifs = []
    for json_path in tqdm(json_files, desc="Extracting bboxes"):
        tif_path = PROC_TIF_DIR / (json_path.stem + ".tif")
        if not tif_path.exists():
            missing_tifs.append(json_path.stem)
            continue

        out_path = BBOX_DIR / (json_path.stem + "_bboxes.json")
        if out_path.exists():
            continue

        bboxes = geo_to_pixel_bboxes(json_path, tif_path)
        with open(out_path, "w") as f:
            json.dump(bboxes, f, indent=2)

    if missing_tifs:
        print(f"  ⚠ No matching TIF found for: {missing_tifs}")
    print(f"  ✓ BBoxes saved → {BBOX_DIR}")


# ═════════════════════════════════════════════
# STAGE 3 — Tile Images + Annotations
# ═════════════════════════════════════════════
def tile_image_and_boxes(
    image:     np.ndarray,
    bboxes:    list[dict],
    tile_size: int   = TILE_SIZE,
    overlap:   float = OVERLAP,
) -> list[tuple[np.ndarray, list[list[int]]]]:
    """
    Slide a window of `tile_size`×`tile_size` over `image` with a stride
    of tile_size × (1 - overlap), yielding sub-images and the subset of
    bboxes visible within each window.

    A bbox is kept in a tile only if the clipped (visible) portion is at
    least MIN_VIS_RATIO of the original box area. This avoids training on
    near-invisible ship slivers at tile edges.

    Only tiles that contain at least one valid bbox are saved — empty
    background tiles are discarded to keep class balance reasonable.

    Returns list of (tile_H×W×4 uint8, [[x1,y1,x2,y2], ...]) tuples.
    """
    H, W   = image.shape[:2]
    stride = int(tile_size * (1 - overlap))
    tiles  = []

    for y in range(0, H - tile_size + 1, stride):
        for x in range(0, W - tile_size + 1, stride):
            tile      = image[y : y + tile_size, x : x + tile_size]
            tile_boxes = []

            for b in bboxes:
                # Intersect box with tile window, shift to tile-local coords
                bx1 = max(b["x_min"], x) - x
                by1 = max(b["y_min"], y) - y
                bx2 = min(b["x_max"], x + tile_size) - x
                by2 = min(b["y_max"], y + tile_size) - y

                if bx2 <= bx1 or by2 <= by1:
                    continue                            # no intersection

                orig_area = ((b["x_max"] - b["x_min"]) *
                             (b["y_max"] - b["y_min"]))
                clip_area = (bx2 - bx1) * (by2 - by1)

                if clip_area / (orig_area + 1e-6) >= MIN_VIS_RATIO:
                    # Store coords + category_id as 5th element
                    tile_boxes.append([int(bx1), int(by1),
                                       int(bx2), int(by2),
                                       int(b.get("category_id", 1))])

            if tile_boxes:                             # discard empty tiles
                tiles.append((tile, tile_boxes))

    return tiles


def stage3_create_tiles():
    """
    Load each normalized TIF + its pixel bbox JSON, run the sliding-window
    tiler, and save:
      - <stem>_tile_<i>.npy   — 4-channel uint8 patch
      - <stem>_tile_<i>.json  — list of [x1,y1,x2,y2] boxes in tile coords

    Returns a dict mapping source image stem → list of tile stem names,
    used later for train/val/test splitting by source image.
    """
    print("\n── Stage 3: Tiling images ───────────────────────────────────")
    tif_files = sorted(PROC_TIF_DIR.glob("*.tif"))
    source_to_tiles: dict[str, list[str]] = {}

    for tif_path in tqdm(tif_files, desc="Tiling"):
        bbox_path = BBOX_DIR / (tif_path.stem + "_bboxes.json")
        if not bbox_path.exists():
            print(f"  ⚠ No bbox file for {tif_path.stem}, skipping.")
            continue

        img = load_rgbn(tif_path)                     # H×W×4 uint8
        with open(bbox_path) as f:
            bboxes = json.load(f)

        if not bboxes:
            print(f"  ⚠ No valid bboxes for {tif_path.stem}, skipping.")
            continue

        tiles = tile_image_and_boxes(img, bboxes)
        tile_stems = []

        for i, (tile_img, tile_boxes) in enumerate(tiles):
            stem     = f"{tif_path.stem}_tile_{i:04d}"
            npy_path = TILE_DIR / f"{stem}.npy"
            box_path = TILE_DIR / f"{stem}.json"

            if not npy_path.exists():
                np.save(npy_path, tile_img)
            if not box_path.exists():
                with open(box_path, "w") as f:
                    json.dump(tile_boxes, f)

            tile_stems.append(stem)

        source_to_tiles[tif_path.stem] = tile_stems

    total_tiles = sum(len(v) for v in source_to_tiles.values())
    print(f"  ✓ {total_tiles} tiles saved → {TILE_DIR}")
    return source_to_tiles


# ═════════════════════════════════════════════
# STAGE 4 — Augmentation (8× per tile)
# ═════════════════════════════════════════════

# All augmentations are bbox-aware via albumentations BboxParams.
# We use pascal_voc format ([x_min, y_min, x_max, y_max]) throughout.
# NOTE: A.CLAHE only supports 1/3-channel images and is excluded here.
# Per-channel CLAHE is applied manually in augment_tile() after this transform.
AUG_TRANSFORM = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.3),                           # swap H and W axes
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.4),      # sensor noise
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),     # motion / focus blur
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(16, 64),
            hole_width_range=(16, 64),
            p=0.3,
        ),                                             # simulate occlusion
        A.RandomScale(scale_limit=0.25, p=0.4),
        A.PadIfNeeded(
            min_height=TILE_SIZE, min_width=TILE_SIZE,
            border_mode=0, p=1.0,                     # pad back to TILE_SIZE
        ),
        A.RandomCrop(height=TILE_SIZE, width=TILE_SIZE, p=1.0),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["labels"],
        min_visibility=0.2,                           # drop boxes < 20% visible post-aug
    ),
)

# Per-channel CLAHE (supports any number of channels; applied with p=0.4)
_CLAHE = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))


def apply_clahe_per_channel(img: np.ndarray, p: float = 0.4) -> np.ndarray:
    """Apply CLAHE independently to each channel of an arbitrary-band uint8 image."""
    if random.random() >= p:
        return img
    out = img.copy()
    for c in range(img.shape[2]):
        out[:, :, c] = _CLAHE.apply(img[:, :, c])
    return out


def augment_tile(
    img: np.ndarray,
    boxes: list[list[int]],
    n_copies: int = AUG_COPIES,
) -> list[tuple[np.ndarray, list[list[int]]]]:
    """
    Apply AUG_TRANSFORM `n_copies` times to one tile.
    Each call uses a different random seed so transforms are independent.

    Returns a list of (aug_img H×W×4, aug_boxes [[x1,y1,x2,y2], ...]).
    Drops any result where all boxes were removed by augmentation.
    """
    results = []
    # Separate pixel coords from category_id (5th element)
    coords = [[b[0], b[1], b[2], b[3]] for b in boxes]
    cat_ids = [b[4] if len(b) > 4 else 1 for b in boxes]

    for _ in range(n_copies):
        augmented  = AUG_TRANSFORM(image=img, bboxes=coords, labels=cat_ids)
        aug_img    = augmented["image"]
        aug_img    = apply_clahe_per_channel(aug_img)  # per-channel CLAHE (4-band safe)
        aug_coords = augmented["bboxes"]
        aug_labels = augmented["labels"]
        # Re-attach category_id as 5th element
        aug_boxes  = [list(map(int, b)) + [int(l)]
                      for b, l in zip(aug_coords, aug_labels)]

        if aug_boxes:                                  # keep only if boxes survive
            results.append((aug_img, aug_boxes))

    return results


def stage4_augment_tiles(source_to_tiles: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    For every raw tile in TILE_DIR, produce AUG_COPIES augmented versions
    saved to AUG_DIR as:
      <stem>_aug_<k>.npy  +  <stem>_aug_<k>.json

    Updates source_to_tiles so each source image now maps to BOTH raw tile
    stems AND their augmented variants — used later for the split.

    Returns updated source_to_tiles.
    """
    print("\n── Stage 4: Augmenting tiles (8× each) ─────────────────────")
    updated: dict[str, list[str]] = {}

    for src_stem, tile_stems in tqdm(
        source_to_tiles.items(), desc="Augmenting"
    ):
        all_stems = []

        for tile_stem in tile_stems:
            npy_path = TILE_DIR / f"{tile_stem}.npy"
            box_path = TILE_DIR / f"{tile_stem}.json"

            img    = np.load(npy_path)
            with open(box_path) as f:
                boxes = json.load(f)

            # Copy the original (un-augmented) tile to AUG_DIR too so the
            # final dataset lives in one place.
            orig_npy = AUG_DIR / f"{tile_stem}.npy"
            orig_box = AUG_DIR / f"{tile_stem}.json"
            if not orig_npy.exists():
                shutil.copy(npy_path, orig_npy)
                shutil.copy(box_path, orig_box)
            all_stems.append(tile_stem)

            # Generate augmented copies
            aug_results = augment_tile(img, boxes)
            for k, (aug_img, aug_boxes) in enumerate(aug_results):
                aug_stem = f"{tile_stem}_aug_{k:02d}"
                aug_npy  = AUG_DIR / f"{aug_stem}.npy"
                aug_box  = AUG_DIR / f"{aug_stem}.json"

                if not aug_npy.exists():
                    np.save(aug_npy, aug_img)
                    with open(aug_box, "w") as f:
                        json.dump(aug_boxes, f)

                all_stems.append(aug_stem)

        updated[src_stem] = all_stems

    total = sum(len(v) for v in updated.values())
    print(f"  ✓ {total} total samples (raw + augmented) → {AUG_DIR}")
    return updated


# ═════════════════════════════════════════════
# STAGE 5 — Train / Val / Test Split
# ═════════════════════════════════════════════
def split_by_source_image(
    source_to_tiles: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    Randomly assign SOURCE images (not individual tiles) to train/val/test.

    ⚠ Splitting by tile would cause data leakage: overlapping tiles from
    the same image would appear in both train and val, inflating metrics.
    Splitting by source image prevents this entirely.

    Returns dict: {"train": [...stems], "val": [...stems], "test": [...stems]}
    """
    source_images = list(source_to_tiles.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(source_images)

    n         = len(source_images)
    n_train   = max(1, int(n * TRAIN_RATIO))
    n_val     = max(1, int(n * VAL_RATIO))
    # remaining goes to test (at least 1 if n >= 3)

    train_srcs = source_images[:n_train]
    val_srcs   = source_images[n_train : n_train + n_val]
    test_srcs  = source_images[n_train + n_val :]

    split_stems: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for src in train_srcs:
        split_stems["train"].extend(source_to_tiles[src])
    for src in val_srcs:
        split_stems["val"].extend(source_to_tiles[src])
    for src in test_srcs:
        split_stems["test"].extend(source_to_tiles[src])

    print("\n── Stage 5: Dataset split (by source image) ─────────────────")
    for split, stems in split_stems.items():
        print(f"  {split:5s}: {len(stems):5d} samples "
              f"({len(stems)/max(1,sum(len(v) for v in split_stems.values()))*100:.1f}%)")
    return split_stems


# ═════════════════════════════════════════════
# STAGE 6 — Export COCO JSON
# ═════════════════════════════════════════════
def export_coco_split(stems: list[str], split: str):
    """
    Copy .npy files for `split` into DATASET_DIR/<split>/ and write a
    COCO-format JSON annotation file at DATASET_DIR/<split>.json.

    COCO bbox format: [x, y, width, height]  ← note: NOT x1,y1,x2,y2.
    We convert from pascal_voc [x1,y1,x2,y2] here.

    Only augmented copies go into 'train'; val and test use originals only
    (stems without '_aug_' suffix) to keep evaluation uncontaminated.
    """
    split_dir = DATASET_DIR / split
    coco = {
        "info":        {"description": f"RT-DETRv2 RGBN Ship Dataset — {split}"},
        "categories":  CATEGORIES,
        "images":      [],
        "annotations": [],
    }
    ann_id  = 1
    img_id  = 0

    # For val / test: use only non-augmented tiles (clean evaluation)
    if split in ("val", "test"):
        stems = [s for s in stems if "_aug_" not in s]

    for stem in tqdm(stems, desc=f"Exporting {split}"):
        src_npy = AUG_DIR / f"{stem}.npy"
        src_box = AUG_DIR / f"{stem}.json"

        if not src_npy.exists() or not src_box.exists():
            print(f"  ⚠ Missing file for stem {stem}, skipping.")
            continue

        # Copy .npy to split directory
        dst_npy = split_dir / f"{stem}.npy"
        if not dst_npy.exists():
            shutil.copy(src_npy, dst_npy)

        with open(src_box) as f:
            boxes = json.load(f)

        img    = np.load(src_npy)
        h, w   = img.shape[:2]

        coco["images"].append({
            "id":        img_id,
            "file_name": f"{stem}.npy",
            "height":    h,
            "width":     w,
        })

        for box in boxes:
            # box may be list [x1,y1,x2,y2,cat_id] OR dict {x_min,...,category_id}
            if isinstance(box, dict):
                x1, y1, x2, y2 = (box["x_min"], box["y_min"],
                                   box["x_max"], box["y_max"])
                cat_id = int(box.get("category_id", 1))
            else:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                cat_id = int(box[4]) if len(box) > 4 else 1

            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue

            coco["annotations"].append({
                "id":          ann_id,
                "image_id":    img_id,
                "category_id": cat_id,             # real class label
                "bbox":        [x1, y1, bw, bh],   # COCO: [x, y, w, h]
                "area":        bw * bh,
                "iscrowd":     0,
            })
            ann_id += 1

        img_id += 1

    out_json = DATASET_DIR / f"{split}.json"
    with open(out_json, "w") as f:
        json.dump(coco, f)

    print(f"  ✓ {split}: {img_id} images, {ann_id-1} annotations → {out_json}")


def stage6_export_coco(split_stems: dict[str, list[str]]):
    print("\n── Stage 6: Exporting COCO JSON ─────────────────────────────")
    for split, stems in split_stems.items():
        export_coco_split(stems, split)


# ═════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  RT-DETRv2 Data Preparation — RGBN 4-Channel Pipeline")
    print("=" * 60)

    make_dirs()
    stage1_normalize_tifs()
    stage2_extract_bboxes()
    source_to_tiles = stage3_create_tiles()
    source_to_tiles = stage4_augment_tiles(source_to_tiles)
    split_stems     = split_by_source_image(source_to_tiles)
    stage6_export_coco(split_stems)

    print("\n" + "=" * 60)
    print("  ✅ Pipeline complete.")
    print(f"  Dataset ready at: {DATASET_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()