"""Measure sparse reconstruction error against the source video.

This mirrors the WebGL shader: the immutable frame-0 base is combined with
the full-resolution keyframe warped by the quantized optical-flow map. The
source MP4 is read only by this offline check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def smoothstep(low, high, value):
    t = np.clip((value - low) / (high - low), 0, 1)
    return t * t * (3 - 2 * t)


def gradient_energy(rgb):
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] +
            0.114 * rgb[..., 2])
    return float(np.mean(np.abs(gray[:, 1:] - gray[:, :-1])) +
                 np.mean(np.abs(gray[1:, :] - gray[:-1, :])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--assets", type=Path, default=Path("public/assets"))
    ap.add_argument("--samples", type=int, default=15)
    ap.add_argument("--out", type=Path, default=Path("analysis/quality.json"))
    args = ap.parse_args()

    meta = json.loads((args.assets / "timeline.json").read_text(encoding="utf-8"))
    rf = meta["referenceFrames"]
    width, height = meta["width"], meta["height"]
    base = np.asarray(Image.open(args.assets / meta["base"]["file"]).convert("RGB"),
                      np.float32)
    mask = np.asarray(Image.open(args.assets / meta["mask"]["file"]).convert("L"),
                      np.float32) / 255.0
    alpha = smoothstep(0.055, 0.145, mask)[..., None].astype(np.float32)
    keys = Image.open(args.assets / rf["keyframeFile"]).convert("RGB")
    flow = Image.open(args.assets / rf["flowFile"]).convert("RGB")
    keys.load()
    flow.load()
    cell_w, cell_h = rf["flowCellWidth"], rf["flowCellHeight"]
    scale = rf["flowScale"]
    offset = rf["flowOffset"]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32),
                                 np.arange(height, dtype=np.float32))
    locked = alpha[..., 0] == 0
    indices = np.unique(np.linspace(
        0, meta["frameCount"] - 1, min(args.samples, meta["frameCount"]))
        .round().astype(int))
    wanted = set(indices.tolist())

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    first_locked = None
    results = []
    frame_number = 0
    while frame_number <= int(indices[-1]):
        ok, bgr = cap.read()
        if not ok:
            break
        if frame_number in wanted:
            reference = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            col = frame_number % rf["flowCols"]
            row = frame_number // rf["flowCols"]
            packed = np.asarray(flow.crop((col * cell_w, row * cell_h,
                                           (col + 1) * cell_w, (row + 1) * cell_h)),
                                np.float32)
            dx = cv2.resize((packed[..., 0] - offset) * scale, (width, height),
                            interpolation=cv2.INTER_LINEAR)
            dy = cv2.resize((packed[..., 1] - offset) * scale, (width, height),
                            interpolation=cv2.INTER_LINEAR)
            anchor = int(rf["anchorKeyframe"][frame_number])
            kx, ky = anchor % rf["keyframeCols"], anchor // rf["keyframeCols"]
            key = np.asarray(keys.crop((kx * width, ky * height,
                                        (kx + 1) * width, (ky + 1) * height)))
            warped = cv2.remap(key, grid_x + dx, grid_y + dy, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
            output = base * (1 - alpha) + warped * alpha
            error = np.abs(output - reference)
            mse = np.mean((output - reference) ** 2)
            if first_locked is None:
                first_locked = warped[locked]
            results.append({
                "frame": frame_number,
                "time": round(frame_number / meta["fps"], 6),
                "anchorFrame": rf["keyframes"][anchor],
                "anchorDistance": abs(frame_number - rf["keyframes"][anchor]),
                "mae": round(float(error.mean()), 5),
                "p95": round(float(np.percentile(error, 95)), 5),
                "psnr": round(float(10 * np.log10(255.0 ** 2 / max(mse, 1e-8))), 3),
                "sharpness": round(gradient_energy(np.clip(output, 0, 255)), 4),
                "referenceSharpness": round(gradient_energy(reference), 4),
                "lockedReferenceDrift": round(float(np.abs(
                    warped[locked] - first_locked).mean()), 5),
            })
        frame_number += 1
    cap.release()

    summary = {
        "sourceFrames": meta["frameCount"],
        "fps": meta["fps"],
        "sampledFrames": len(results),
        "width": width,
        "height": height,
        "strategy": rf["strategy"],
        "keyframes": len(rf["keyframes"]),
        "flowFrames": rf["frames"],
        "fullResolution": rf["fullResolution"],
        "shaderLockedPixelRatio": round(float(locked.mean()), 5),
        "meanMae": round(float(np.mean([r["mae"] for r in results])), 5),
        "maxMae": round(float(np.max([r["mae"] for r in results])), 5),
        "meanPsnr": round(float(np.mean([r["psnr"] for r in results])), 3),
        "meanSharpness": round(float(np.mean(
            [r["sharpness"] for r in results])), 4),
        "referenceSharpness": round(float(np.mean(
            [r["referenceSharpness"] for r in results])), 4),
        "meanLockedReferenceDrift": round(float(np.mean(
            [r["lockedReferenceDrift"] for r in results])), 5),
        "baseTextureUpdates": 0,
        "frames": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "frames"},
                     indent=2))


if __name__ == "__main__":
    main()
