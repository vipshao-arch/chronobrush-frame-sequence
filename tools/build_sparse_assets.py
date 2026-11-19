"""Build a compact, full-resolution blended-keyframe reconstruction dataset.

The source video is read only here. Runtime assets contain full-resolution
WebP keyframes, two-way optical-flow maps, a locked base, a motion mask and
the existing drawing-command timeline. Each intermediate frame blends a warp
from its previous keyframe with a warp from its next keyframe, so the
reference frame changes continuously at keyframe boundaries. No MP4 or video
decoder is shipped.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ASSETS = ROOT / "public" / "assets"
FLOW_CELL = (368, 158)
FLOW_COLS = 13
KEY_COLS = 4
FLOW_SCALE = 0.5
FLOW_OFFSET = 128
KEYFRAME_STEP = 8


def save_webp(rgb: np.ndarray, path: Path, *, lossless=False,
              quality=98, method=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, "WEBP", lossless=lossless,
                              quality=quality, method=method)


def read_video(video: Path, width: int, height: int):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    full_frames = []
    small_frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        expected = (height, width, 3)
        if rgb.shape != expected:
            raise ValueError(f"Unexpected frame size {rgb.shape}, expected {expected}")
        full_frames.append(rgb)
        small_frames.append(cv2.resize(rgb, FLOW_CELL,
                                       interpolation=cv2.INTER_AREA))
    cap.release()
    if not full_frames:
        raise RuntimeError("No frames decoded")
    return np.stack(full_frames), np.stack(small_frames)


def uniform_keys(count: int, step: int):
    keys = list(range(0, count, step))
    if keys[-1] != count - 1:
        keys.append(count - 1)
    return np.array(keys, dtype=int)


def brackets(keys: np.ndarray, frame: int):
    lo = int(np.clip(np.searchsorted(keys, frame, side="right") - 1,
                     0, len(keys) - 2))
    hi = lo + 1
    t = float(np.clip((frame - keys[lo]) / (keys[hi] - keys[lo]), 0, 1))
    return lo, hi, t


def make_key_atlas(frames: np.ndarray, keys: np.ndarray, width: int,
                   height: int):
    rows = math.ceil(len(keys) / KEY_COLS)
    sheet = np.zeros((rows * height, KEY_COLS * width, 3), np.uint8)
    for slot, index in enumerate(keys):
        x, y = (slot % KEY_COLS) * width, (slot // KEY_COLS) * height
        sheet[y:y + height, x:x + width] = frames[index]
    return sheet, rows


def quantized_flow(current: np.ndarray, target: np.ndarray,
                   width: int, height: int):
    a = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    # Encode full-canvas pixel displacement. The shader adds this directly
    # to uv * u_resolution before sampling the full-resolution keyframe.
    flow = cv2.calcOpticalFlowFarneback(a, b, None,
                                       0.5, 3, 15, 3, 5, 1.2, 0)
    flow[..., 0] *= width / current.shape[1]
    flow[..., 1] *= height / current.shape[0]
    qx = np.clip(np.rint(flow[..., 0] / FLOW_SCALE + FLOW_OFFSET), 0, 255)
    qy = np.clip(np.rint(flow[..., 1] / FLOW_SCALE + FLOW_OFFSET), 0, 255)
    return np.dstack([qx, qy, np.full(qx.shape, FLOW_OFFSET)]).astype(np.uint8)


def make_flow_atlas(frames: np.ndarray, keys: np.ndarray,
                    width: int, height: int):
    cell_w, cell_h = FLOW_CELL
    cells = []
    flow_cell0 = []
    flow_cell1 = []
    blend_weight = []
    saturated = 0
    total = 0

    def add_cell(frame: int, key_slot: int):
        nonlocal saturated, total
        key_frame = int(keys[key_slot])
        if key_frame == frame:
            cell = np.full((cell_h, cell_w, 3), FLOW_OFFSET, np.uint8)
        else:
            cell = quantized_flow(frames[frame], frames[key_frame],
                                  width, height)
            before = np.dstack([cell[..., 0], cell[..., 1]])
            saturated += int(np.count_nonzero(
                (before <= 0) | (before >= 255)))
        total += cell_w * cell_h * 2
        cells.append(cell)
        return len(cells) - 1

    for frame in range(len(frames)):
        lo, hi, mix = brackets(keys, frame)
        blend_weight.append(round(mix, 5))
        if mix <= 1e-6:
            only = add_cell(frame, lo)
            flow_cell0.append(only)
            flow_cell1.append(only)
        elif mix >= 1 - 1e-6:
            only = add_cell(frame, hi)
            flow_cell0.append(only)
            flow_cell1.append(only)
        else:
            first = add_cell(frame, lo)
            second = add_cell(frame, hi)
            flow_cell0.append(first)
            flow_cell1.append(second)

    rows = math.ceil(len(cells) / FLOW_COLS)
    sheet = np.full((rows * cell_h, FLOW_COLS * cell_w, 3), 128, np.uint8)
    for index, cell in enumerate(cells):
        x = (index % FLOW_COLS) * cell_w
        y = (index // FLOW_COLS) * cell_h
        sheet[y:y + cell_h, x:x + cell_w] = cell
    info = {
        "rows": rows,
        "cells": len(cells),
        "cellWidth": cell_w,
        "cellHeight": cell_h,
        "flowCell0": flow_cell0,
        "flowCell1": flow_cell1,
        "blendWeight": blend_weight,
        "saturatedChannelRatio": round(saturated / total, 6),
    }
    return sheet, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=SOURCE_ASSETS)
    ap.add_argument("--step", type=int, default=KEYFRAME_STEP)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--flow-quality", type=int, default=92)
    args = ap.parse_args()

    meta = json.loads((SOURCE_ASSETS / "timeline.json").read_text(encoding="utf-8"))
    width, height = meta["width"], meta["height"]
    frames, small = read_video(args.video, width, height)
    if len(frames) != meta["frameCount"]:
        raise ValueError("Video frame count does not match timeline")

    keys = uniform_keys(len(frames), args.step)
    key_sheet, key_rows = make_key_atlas(frames, keys, width, height)
    flow_sheet, flow_info = make_flow_atlas(small, keys, width, height)

    save_webp(key_sheet, args.out / "keyframes.webp",
              quality=args.quality, method=4)
    save_webp(flow_sheet, args.out / "flow.webp",
              quality=args.flow_quality, method=6)

    meta["referenceFrames"] = {
        "sync": "timeline.frameAtTime",
        "frames": len(frames),
        "source": "pre-analyzed WebP keyframes and bidirectional optical flow",
        "strategy": "full-resolution keyframes + blended bidirectional optical flow",
        "keyframeFile": "keyframes.webp",
        "flowFile": "flow.webp",
        "keyframes": keys.tolist(),
        "flowCell0": flow_info["flowCell0"],
        "flowCell1": flow_info["flowCell1"],
        "blendWeight": flow_info["blendWeight"],
        "flowCells": flow_info["cells"],
        "blendMode": "linear",
        "flowEncoding": f"lossy WebP q{args.flow_quality}",
        "keyframeEncoding": f"lossy WebP q{args.quality}",
        "keyframeCols": KEY_COLS,
        "keyframeRows": key_rows,
        "keyframeWidth": width,
        "keyframeHeight": height,
        "flowCols": FLOW_COLS,
        "flowRows": flow_info["rows"],
        "flowCellWidth": flow_info["cellWidth"],
        "flowCellHeight": flow_info["cellHeight"],
        "flowScale": FLOW_SCALE,
        "flowOffset": 128,
        "flowSaturatedChannelRatio": flow_info["saturatedChannelRatio"],
        "fullResolution": True,
    }
    meta.pop("sheet", None)
    (args.out / "timeline.json").write_text(
        json.dumps(meta, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({
        "frames": len(frames),
        "keyframes": len(keys),
        "flowCells": flow_info["cells"],
        "keyAtlas": [key_sheet.shape[1], key_sheet.shape[0]],
        "flowAtlas": [flow_sheet.shape[1], flow_sheet.shape[0]],
        "keyframeQuality": args.quality,
        "flowQuality": args.flow_quality,
        "assetBytes": sum((args.out / name).stat().st_size for name in
                          ["base.webp", "motion-mask.webp",
                           "keyframes.webp", "flow.webp", "timeline.json"]),
    }, indent=2))


if __name__ == "__main__":
    main()
