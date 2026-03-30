"""
RT-DETRv2 RGBN Inference Script
Preprocessing matches prepare.py exactly:
  - Reads 4-band GeoTIFF (R, G, B, NIR) via rasterio
  - Per-band 2nd-98th percentile normalization → uint8
  - Resize to model input size
  - Normalize with ImageNet-style mean/std per channel
  - Tiles large images automatically (same 1024px / 25% overlap logic)

Usage:
    # Single GeoTIFF
    python tools/infer_rgbn.py \
        -c configs/rtdetrv2/rtdetrv2_r50vd_rgbn.yml \
        -r ./logs/best.pth \
        --input path/to/image.tif \
        --output output/infer_results \
        --thresh 0.3

    # Folder of GeoTIFFs
    python tools/infer_rgbn.py \
        -c configs/rtdetrv2/rtdetrv2_r50vd_rgbn.yml \
        -r ./logs/best.pth \
        --input path/to/tif_folder \
        --output output/infer_results \
        --thresh 0.3

    # Already-prepared .npy file (skip tif loading)
    python tools/infer_rgbn.py \
        -c configs/rtdetrv2/rtdetrv2_r50vd_rgbn.yml \
        -r ./logs/best.pth \
        --input path/to/patch.npy \
        --output output/infer_results \
        --thresh 0.3
"""

import argparse
import os
import sys
import glob
import json
import torch
import numpy as np
import cv2

sys.path.insert(0, '.')

# rasterio is only needed for .tif input
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


# ── Category map (matches prepare.py exactly) ────────────────────────────────
CATEGORIES = [
    {"id": 1,  "name": "Boat"},
    {"id": 2,  "name": "Vehicle-FM"},
    {"id": 3,  "name": "Rail Car"},
    {"id": 4,  "name": "Sheds-C"},
    {"id": 5,  "name": "Sheds-T"},
    {"id": 6,  "name": "Ship-L"},
    {"id": 7,  "name": "Ship-S"},
    {"id": 8,  "name": "Swimming Pool"},
    {"id": 9,  "name": "Water Tank"},
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
    {"id": 24, "name": "Wind Turbine"},
]
# 0-indexed for model output (model predicts 0–23)
CATS = {c['id'] - 1: c['name'] for c in CATEGORIES}

# One distinct color per class
COLORS = [
    (255,  56,  56), (255, 157, 151), (255, 112,  31), (255, 178,  29),
    (207, 210,  49), ( 72, 249,  10), (146, 204,  23), ( 61, 219, 134),
    ( 26, 147,  52), (  0, 212, 187), ( 44, 153, 168), (  0, 194, 255),
    ( 52,  69, 147), (100, 115, 255), (  0,  24, 236), (132,  56, 255),
    ( 82,   0, 133), (203,  56, 255), (255, 149, 200), (255,  55, 199),
    (255, 100,   0), (  0, 255,   0), (  0,   0, 255), (255,   0,   0),
]


# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='RT-DETRv2 RGBN Inference')
    p.add_argument('-c', '--config',   required=True,  help='Path to config YAML')
    p.add_argument('-r', '--resume',   required=True,  help='Path to .pth checkpoint')
    p.add_argument('--input',          required=True,  help='.tif / .npy file or folder')
    p.add_argument('--output',         default='output/infer_results')
    p.add_argument('--thresh',         type=float, default=0.3)
    p.add_argument('--input_size',     type=int,   default=640,  help='Model input size')
    p.add_argument('--tile_size',      type=int,   default=1024, help='Tile size for large TIFs')
    p.add_argument('--overlap',        type=float, default=0.25, help='Tile overlap ratio')
    p.add_argument('--device',         default='cuda')
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# Preprocessing — exactly mirrors prepare.py Stage 1
# ═════════════════════════════════════════════════════════════════════════════

def percentile_norm(band: np.ndarray) -> np.ndarray:
    """
    Per-band 2nd–98th percentile stretch → uint8.
    Identical to prepare.py::percentile_norm().
    """
    p2  = np.percentile(band, 2)
    p98 = np.percentile(band, 98)
    stretched = (band - p2) / (p98 - p2 + 1e-6) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)


def load_tif_as_rgbn(tif_path: str) -> np.ndarray:
    """
    Load a 4-band GeoTIFF → H×W×4 uint8 (R, G, B, NIR).
    Identical pipeline to prepare.py::load_rgbn().
    """
    if not HAS_RASTERIO:
        raise ImportError(
            "rasterio is required for .tif input.\n"
            "Install with: pip install rasterio"
        )
    with rasterio.open(tif_path) as src:
        if src.count < 4:
            raise ValueError(
                f"{tif_path} has only {src.count} band(s); expected 4 (R,G,B,NIR)."
            )
        r   = src.read(1).astype(np.float32)
        g   = src.read(2).astype(np.float32)
        b   = src.read(3).astype(np.float32)
        nir = src.read(4).astype(np.float32)

    return np.stack([
        percentile_norm(r),
        percentile_norm(g),
        percentile_norm(b),
        percentile_norm(nir),
    ], axis=-1)   # H×W×4 uint8


def load_npy_as_rgbn(npy_path: str) -> np.ndarray:
    """
    Load an already-prepared .npy tile → H×W×4 uint8.
    Handles both (H,W,4) and (4,H,W) layouts.
    .npy files were saved by prepare.py after percentile_norm so we do NOT
    re-normalise — we just ensure the array is uint8 H×W×4.
    """
    arr = np.load(npy_path)

    # channel-first → channel-last
    if arr.ndim == 3 and arr.shape[0] == 4:
        arr = arr.transpose(1, 2, 0)   # (4,H,W) → (H,W,4)

    if arr.shape[-1] != 4:
        raise ValueError(f"Expected 4 channels, got shape {arr.shape}")

    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32)
    if arr.max() <= 1.0:
        # float [0,1] → uint8
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    else:
        arr = arr.clip(0, 255).astype(np.uint8)

    return arr   # H×W×4 uint8


# ═════════════════════════════════════════════════════════════════════════════
# Tiling — mirrors prepare.py Stage 3
# ═════════════════════════════════════════════════════════════════════════════

def tile_image(img_hwc: np.ndarray, tile_size: int, overlap: float):
    """
    Yield (tile_array H×W×4, row_offset, col_offset).
    Matches prepare.py: tile_size=1024, overlap=0.25 → stride=768.
    Edge tiles are zero-padded to tile_size×tile_size.
    """
    H, W, _ = img_hwc.shape
    stride   = int(tile_size * (1.0 - overlap))

    # Image fits in one tile
    if H <= tile_size and W <= tile_size:
        pad_h = max(0, tile_size - H)
        pad_w = max(0, tile_size - W)
        tile  = np.pad(img_hwc, ((0, pad_h), (0, pad_w), (0, 0))) if (pad_h or pad_w) else img_hwc
        yield tile, 0, 0
        return

    row_starts = list(range(0, max(1, H - tile_size + 1), stride))
    col_starts = list(range(0, max(1, W - tile_size + 1), stride))

    if row_starts[-1] + tile_size < H:
        row_starts.append(max(0, H - tile_size))
    if col_starts[-1] + tile_size < W:
        col_starts.append(max(0, W - tile_size))

    for r in row_starts:
        for c in col_starts:
            tile = img_hwc[r:r + tile_size, c:c + tile_size]
            th, tw = tile.shape[:2]
            if th < tile_size or tw < tile_size:
                tile = np.pad(tile, ((0, tile_size - th), (0, tile_size - tw), (0, 0)))
            yield tile, r, c


# ═════════════════════════════════════════════════════════════════════════════
# Model preprocessing
# ═════════════════════════════════════════════════════════════════════════════

# Must match values used during training
MEAN = np.array([0.485, 0.456, 0.406, 0.45],  dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225, 0.22],  dtype=np.float32)


def preprocess_tile(tile_hwc: np.ndarray, input_size: int, device):
    """
    tile_hwc: H×W×4 uint8
    Returns: (1, 4, input_size, input_size) float32 tensor on device
    """
    # uint8 [0,255] → float [0,1], channel-first
    tile_chw = tile_hwc.transpose(2, 0, 1).astype(np.float32) / 255.0  # (4,H,W)

    # Resize each channel to model input size
    resized = np.stack([
        cv2.resize(tile_chw[c], (input_size, input_size),
                   interpolation=cv2.INTER_LINEAR)
        for c in range(4)
    ], axis=0)   # (4, input_size, input_size)

    # ImageNet-style normalisation
    resized = (resized - MEAN.reshape(4, 1, 1)) / STD.reshape(4, 1, 1)

    return torch.from_numpy(resized).unsqueeze(0).float().to(device)


# ═════════════════════════════════════════════════════════════════════════════
# Inference
# ═════════════════════════════════════════════════════════════════════════════

def run_tile(model, tile_hwc, input_size, device, thresh):
    """
    Returns list of (x1, y1, x2, y2, score, class_id) in tile pixel coords.
    """
    H, W  = tile_hwc.shape[:2]
    tensor = preprocess_tile(tile_hwc, input_size, device)

    with torch.no_grad():
        outputs = model(tensor)

    logits = outputs['pred_logits'][0]   # (num_queries, num_classes)
    boxes  = outputs['pred_boxes'][0]    # (num_queries, 4) cxcywh norm [0,1]

    scores, class_ids = logits.sigmoid().max(-1)
    keep = scores > thresh

    dets = []
    for score, cls, box in zip(scores[keep], class_ids[keep], boxes[keep]):
        cx, cy, bw, bh = box.tolist()
        x1 = (cx - bw / 2) * W
        y1 = (cy - bh / 2) * H
        x2 = (cx + bw / 2) * W
        y2 = (cy + bh / 2) * H
        dets.append((x1, y1, x2, y2, score.item(), cls.item()))

    return dets


# ═════════════════════════════════════════════════════════════════════════════
# Visualisation
# ═════════════════════════════════════════════════════════════════════════════

def draw_boxes(img_bgr, detections):
    for x1, y1, x2, y2, score, cls in detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = COLORS[int(cls) % len(COLORS)]
        label = CATS.get(int(cls), str(int(cls)))
        text  = f'{label} {score:.2f}'
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img_bgr, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(img_bgr, text, (x1 + 1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img_bgr


def make_vis(img_hwc, detections, max_display=2048):
    """
    Build side-by-side RGB+predictions / NIR visualization.
    Downscales large images to max_display pixels on the long side.
    """
    H, W  = img_hwc.shape[:2]
    scale = min(1.0, max_display / max(H, W))
    dw, dh = int(W * scale), int(H * scale)

    rgb_bgr = cv2.cvtColor(img_hwc[:, :, :3], cv2.COLOR_RGB2BGR)
    rgb_bgr = cv2.resize(rgb_bgr, (dw, dh))

    scaled_dets = [
        (x1 * scale, y1 * scale, x2 * scale, y2 * scale, s, c)
        for x1, y1, x2, y2, s, c in detections
    ]
    rgb_bgr = draw_boxes(rgb_bgr, scaled_dets)

    nir = img_hwc[:, :, 3]
    nir = cv2.resize(nir, (dw, dh))
    nir = cv2.normalize(nir, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    nir_bgr = cv2.cvtColor(nir, cv2.COLOR_GRAY2BGR)

    cv2.putText(rgb_bgr, 'RGB + Predictions', (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(nir_bgr, 'NIR channel',       (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return np.hstack([rgb_bgr, nir_bgr])


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    # Load model
    from src.core import YAMLConfig
    cfg   = YAMLConfig(args.config)
    model = cfg.model.eval().to(device)
    ckpt  = torch.load(args.resume, map_location=device)
    model.load_state_dict(ckpt['model'])
    print(f'Checkpoint : {args.resume}')
    print(f'Threshold  : {args.thresh}')
    print(f'Tile size  : {args.tile_size}  Overlap: {args.overlap}')
    print(f'Input size : {args.input_size}\n')

    # Collect files
    if os.path.isdir(args.input):
        files = (sorted(glob.glob(os.path.join(args.input, '*.tif')))  +
                 sorted(glob.glob(os.path.join(args.input, '*.tiff'))) +
                 sorted(glob.glob(os.path.join(args.input, '*.npy'))))
    else:
        files = [args.input]

    if not files:
        print(f'No .tif / .npy files found in: {args.input}')
        return

    print(f'Files to process: {len(files)}\n{"─"*60}')

    for fpath in files:
        fname = os.path.splitext(os.path.basename(fpath))[0]
        ext   = os.path.splitext(fpath)[1].lower()
        print(f'\n▶  {os.path.basename(fpath)}')

        # ── Load using same method as prepare.py ────────────────────────
        if ext in ('.tif', '.tiff'):
            img_hwc = load_tif_as_rgbn(fpath)     # percentile_norm applied
        elif ext == '.npy':
            img_hwc = load_npy_as_rgbn(fpath)     # already normed by prepare.py
        else:
            print(f'   Skipping unsupported extension: {ext}')
            continue

        H, W = img_hwc.shape[:2]
        print(f'   Size: {W}×{H}')

        # ── Tile → infer → collect full-image coords ─────────────────────
        all_dets = []
        tiles    = list(tile_image(img_hwc, args.tile_size, args.overlap))
        print(f'   Tiles: {len(tiles)}')

        for tile_hwc, row_off, col_off in tiles:
            dets = run_tile(model, tile_hwc, args.input_size, device, args.thresh)
            for x1, y1, x2, y2, score, cls in dets:
                all_dets.append((
                    x1 + col_off, y1 + row_off,
                    x2 + col_off, y2 + row_off,
                    score, cls
                ))

        print(f'   Detections: {len(all_dets)}')

        # ── Print to terminal ────────────────────────────────────────────
        for i, (x1, y1, x2, y2, score, cls) in enumerate(all_dets):
            label = CATS.get(int(cls), str(int(cls)))
            print(f'   [{i+1:3d}] {label:15s}  {score:.3f}  '
                  f'({x1:.0f},{y1:.0f}) → ({x2:.0f},{y2:.0f})')

        # ── Save visualisation ───────────────────────────────────────────
        vis      = make_vis(img_hwc[:H, :W], all_dets)
        out_path = os.path.join(args.output, f'{fname}_{len(all_dets)}dets.jpg')
        cv2.imwrite(out_path, vis)
        print(f'   Saved → {out_path}')

        # ── Save detections as JSON ──────────────────────────────────────
        json_path = os.path.join(args.output, f'{fname}_dets.json')
        with open(json_path, 'w') as f:
            json.dump([{
                'bbox_xyxy': [round(x1, 1), round(y1, 1),
                              round(x2, 1), round(y2, 1)],
                'score':     round(score, 4),
                'class_id':  int(cls),
                'class_name': CATS.get(int(cls), str(int(cls))),
            } for x1, y1, x2, y2, score, cls in all_dets], f, indent=2)
        print(f'   JSON  → {json_path}')

    print(f'\n{"─"*60}\nDone. Results in: {args.output}/')


if __name__ == '__main__':
    main()