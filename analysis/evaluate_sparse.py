"""Evaluate one sparse-keyframe candidate used during asset selection.

The runtime candidate uses only a few full-resolution WebP keyframes and a
low-resolution flow sheet for all 145 frames. This script measures the true
error against the original video, including WebP and flow quantization. It
also compares against the legacy half-resolution frame sheet, which is used
only as an offline baseline and is not part of the runtime asset package.
"""

from __future__ import annotations

import argparse
import json
import math
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "public" / "assets"
FLOW_CELL = (368, 158)
FLOW_COLS = 13
KEY_COLS = 4
FLOW_SCALE = 0.5
FLOW_OFFSET = 128.0
LEGACY_SHEET = ROOT / "analysis" / "legacy-reference.webp"


def read_small_frames(video: Path):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(video)
    grays = []
    colors = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, FLOW_CELL, interpolation=cv2.INTER_AREA)
        colors.append(small)
        grays.append(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY))
    cap.release()
    return np.stack(grays), np.stack(colors)


def read_full_frames(video: Path, wanted: set[int]):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(video)
    frames = {}
    index = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if index in wanted:
            frames[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        index += 1
    cap.release()
    return frames, index


def uniform_keys(count: int, step: int):
    keys = list(range(0, count, step))
    if keys[-1] != count - 1:
        keys.append(count - 1)
    return np.array(keys, dtype=int)


def adaptive_keys(scores: np.ndarray, target: int, max_gap: int = 8):
    n = len(scores) + 1

    def build(threshold):
        keys = [0]
        accumulated = 0.0
        for i in range(1, n):
            accumulated += float(scores[i - 1])
            if accumulated >= threshold or i - keys[-1] >= max_gap:
                keys.append(i)
                accumulated = 0.0
        if n - 1 not in keys:
            if n - 1 - keys[-1] <= max_gap // 2:
                keys[-1] = n - 1
            else:
                keys.append(n - 1)
        return np.array(keys, dtype=int)

    lo, hi = 0.0, float(scores.sum())
    for _ in range(40):
        mid = (lo + hi) / 2
        if len(build(mid)) > target:
            lo = mid
        else:
            hi = mid
    return build(hi)


def nearest_anchor_indices(keys: np.ndarray, count: int):
    distance = np.abs(keys[None, :] - np.arange(count)[:, None])
    return distance.argmin(1).astype(int)


def encode_key_atlas(keys: np.ndarray, full: dict[int, np.ndarray],
                     width: int, height: int, quality: int):
    rows = math.ceil(len(keys) / KEY_COLS)
    sheet = Image.new("RGB", (KEY_COLS * width, rows * height))
    for i, frame in enumerate(keys):
        x, y = (i % KEY_COLS) * width, (i // KEY_COLS) * height
        sheet.paste(Image.fromarray(full[int(frame)]), (x, y))
    buffer = BytesIO()
    sheet.save(buffer, "WEBP", quality=quality, method=4)
    data = buffer.getvalue()
    buffer.seek(0)
    decoded = Image.open(buffer).convert("RGB")
    decoded.load()
    return data, decoded, KEY_COLS, rows


def legacy_reference_image():
    """Decode the legacy 735x315 frame sheet from Git into memory."""
    import subprocess

    if not LEGACY_SHEET.exists():
        blob = subprocess.check_output(
            ["git", "show", "a9c9e43:public/assets/reference.webp"])
        LEGACY_SHEET.write_bytes(blob)
    image = Image.open(LEGACY_SHEET).convert("RGB")
    image.load()
    return image


def encode_flow_sheet(grays: np.ndarray, keys: np.ndarray, anchors: np.ndarray,
                      width: int, height: int):
    n, cell_h, cell_w = grays.shape
    rows = math.ceil(n / FLOW_COLS)
    sheet = np.full((rows * cell_h, FLOW_COLS * cell_w, 3), 128, np.uint8)
    scale_x = width / cell_w
    scale_y = height / cell_h
    saturated = 0
    total = 0
    for frame in range(n):
        anchor_index = int(anchors[frame])
        anchor_frame = int(keys[anchor_index])
        if anchor_frame == frame:
            flow = np.zeros((cell_h, cell_w, 2), np.float32)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                grays[frame], grays[anchor_frame], None,
                0.5, 3, 15, 3, 5, 1.2, 0)
            flow[..., 0] *= scale_x
            flow[..., 1] *= scale_y
        qx = np.clip(np.rint(flow[..., 0] / FLOW_SCALE + FLOW_OFFSET), 0, 255)
        qy = np.clip(np.rint(flow[..., 1] / FLOW_SCALE + FLOW_OFFSET), 0, 255)
        # Blue is a constant neutral pixel to keep the lossless WebP compact.
        qz = np.full(qx.shape, FLOW_OFFSET, np.float64)
        saturated += int(np.count_nonzero((qx <= 0) | (qx >= 255) |
                                          (qy <= 0) | (qy >= 255)))
        total += qx.size * 3
        cell = np.dstack([qx.astype(np.uint8), qy.astype(np.uint8),
                          qz.astype(np.uint8)])
        x, y = (frame % FLOW_COLS) * cell_w, (frame // FLOW_COLS) * cell_h
        sheet[y:y + cell_h, x:x + cell_w] = cell
    buffer = BytesIO()
    Image.fromarray(sheet).save(buffer, "WEBP", lossless=True, method=6)
    data = buffer.getvalue()
    buffer.seek(0)
    decoded = Image.open(buffer).convert("RGB")
    decoded.load()
    return data, decoded, FLOW_COLS, rows, cell_w, cell_h, saturated / total


def smoothstep(low, high, value):
    t = np.clip((value - low) / (high - low), 0, 1)
    return t * t * (3 - 2 * t)


def gradient_energy(rgb: np.ndarray):
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] +
            0.114 * rgb[..., 2])
    return float(np.mean(np.abs(gray[:, 1:] - gray[:, :-1])) +
                 np.mean(np.abs(gray[1:, :] - gray[:-1, :])))


def metrics(reference: np.ndarray, output: np.ndarray):
    error = np.abs(output.astype(np.float32) - reference.astype(np.float32))
    mse = np.mean((output.astype(np.float32) - reference.astype(np.float32)) ** 2)
    psnr = 10 * np.log10(255.0 ** 2 / max(mse, 1e-8))
    return {
        "mae": round(float(error.mean()), 4),
        "p95": round(float(np.percentile(error, 95)), 4),
        "psnr": round(float(psnr), 3),
        "sharpness": round(gradient_energy(np.clip(output, 0, 255)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path,
                    default=Path(r"C:\Users\vipshao\Desktop\参考 (2).mp4"))
    ap.add_argument("--mode", choices=["uniform", "adaptive"], default="adaptive")
    ap.add_argument("--step", type=int, default=6)
    ap.add_argument("--target", type=int, default=25)
    ap.add_argument("--quality", type=int, default=98)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    meta = json.loads((ASSETS / "timeline.json").read_text(encoding="utf-8"))
    width, height = meta["width"], meta["height"]
    base = np.asarray(Image.open(ASSETS / "base.webp").convert("RGB"), np.float32)
    mask = np.asarray(Image.open(ASSETS / "motion-mask.webp").convert("L"), np.float32) / 255.0
    alpha = smoothstep(0.055, 0.145, mask)[..., None].astype(np.float32)
    lowres = legacy_reference_image()
    low_cols = 7
    low_cw, low_ch = 735, 315

    grays, colors = read_small_frames(args.video)
    n = len(grays)
    scores = np.abs(colors[1:].astype(np.float32) -
                    colors[:-1].astype(np.float32)).mean((1, 2, 3))
    keys = (uniform_keys(n, args.step) if args.mode == "uniform"
            else adaptive_keys(scores, args.target))
    anchors = nearest_anchor_indices(keys, n)
    samples = np.unique(np.linspace(0, n - 1, 15).round().astype(int))
    wanted = set(map(int, keys)) | set(map(int, samples))
    full, _ = read_full_frames(args.video, wanted)

    key_bytes, key_image, key_cols, key_rows = encode_key_atlas(
        keys, full, width, height, args.quality)
    flow_bytes, flow_image, flow_cols, flow_rows, flow_cw, flow_ch, saturation = (
        encode_flow_sheet(grays, keys, anchors, width, height))

    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32),
                                 np.arange(height, dtype=np.float32))
    results = []
    for frame in samples:
        frame = int(frame)
        reference = full[frame].astype(np.float32)

        # Current low-resolution frame-sheet release.
        lr_x, lr_y = (frame % low_cols) * low_cw, (frame // low_cols) * low_ch
        lr_cell = np.asarray(lowres.crop((lr_x, lr_y, lr_x + low_cw,
                                          lr_y + low_ch)))
        lr_up = cv2.resize(lr_cell, (width, height),
                           interpolation=cv2.INTER_LINEAR).astype(np.float32)
        low_output = base * (1 - alpha) + lr_up * alpha

        # Sparse full-resolution keyframe warped by the quantized flow map.
        fx, fy = (frame % flow_cols) * flow_cw, (frame // flow_cols) * flow_ch
        flow_cell = np.asarray(flow_image.crop((fx, fy, fx + flow_cw,
                                                fy + flow_ch))).astype(np.float32)
        dx = (flow_cell[..., 0] - FLOW_OFFSET) * FLOW_SCALE
        dy = (flow_cell[..., 1] - FLOW_OFFSET) * FLOW_SCALE
        map_x = grid_x + cv2.resize(dx, (width, height),
                                    interpolation=cv2.INTER_LINEAR)
        map_y = grid_y + cv2.resize(dy, (width, height),
                                    interpolation=cv2.INTER_LINEAR)
        anchor = int(anchors[frame])
        kx, ky = (anchor % key_cols) * width, (anchor // key_cols) * height
        key_frame = np.asarray(key_image.crop((kx, ky, kx + width,
                                               ky + height)))
        warped = cv2.remap(key_frame, map_x.astype(np.float32),
                           map_y.astype(np.float32), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        sparse_output = base * (1 - alpha) + warped * alpha

        results.append({
            "frame": frame,
            "anchorFrame": int(keys[anchor]),
            "anchorDistance": int(abs(frame - keys[anchor])),
            "lowres": metrics(reference, low_output),
            "sparse": metrics(reference, sparse_output),
            "referenceSharpness": round(gradient_energy(reference), 4),
        })

    summary = {
        "mode": args.mode,
        "step": args.step,
        "quality": args.quality,
        "frames": n,
        "keyframeCount": len(keys),
        "keyframes": keys.tolist(),
        "maxAnchorDistance": max(r["anchorDistance"] for r in results),
        "keyAtlasBytes": len(key_bytes),
        "flowAtlasBytes": len(flow_bytes),
        "flowSaturatedChannelRatio": round(saturation, 6),
        "lowresMeanMae": round(float(np.mean([r["lowres"]["mae"] for r in results])), 4),
        "sparseMeanMae": round(float(np.mean([r["sparse"]["mae"] for r in results])), 4),
        "lowresMeanPsnr": round(float(np.mean([r["lowres"]["psnr"] for r in results])), 3),
        "sparseMeanPsnr": round(float(np.mean([r["sparse"]["psnr"] for r in results])), 3),
        "lowresMeanSharpness": round(float(np.mean([r["lowres"]["sharpness"] for r in results])), 4),
        "sparseMeanSharpness": round(float(np.mean([r["sparse"]["sharpness"] for r in results])), 4),
        "referenceMeanSharpness": round(float(np.mean([r["referenceSharpness"] for r in results])), 4),
        "samples": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in
                      ("keyframes", "samples")}, indent=2))


if __name__ == "__main__":
    main()
