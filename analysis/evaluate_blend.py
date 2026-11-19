"""Evaluate compact blended-keyframe reconstruction candidates.

Runtime assets keep full-resolution WebP keyframes and one compact
optical-flow atlas. Frames between keyframes store two flow cells: current
frame -> previous key and current frame -> next key. The shader blends the two
warped keyframes with a continuous temporal weight. Exact keyframes reuse one
neutral flow cell, which keeps the atlas compact without an anchor switch.

The source MP4 is read only by this offline evaluation.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "public" / "assets"
DEFAULT_VIDEO = Path(r"C:\Users\vipshao\Desktop\参考 (2).mp4")
KEY_COLS = 4
FLOW_COLS = 13
FLOW_OFFSET = 128.0


def read_video(video: Path, wanted_full: set[int], cell_size: tuple[int, int]):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    small = []
    full = {}
    index = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small.append(cv2.resize(rgb, cell_size, interpolation=cv2.INTER_AREA))
        if index in wanted_full:
            full[index] = rgb
        index += 1
    cap.release()
    return np.stack(small), full, index


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


def blend_weight(candidate, t):
    """Temporal keyframe blend weight.

    linear and smoothstep are continuous across adjacent keyframe intervals.
    nearest reproduces the old hard anchor switch. band is an experimental
    piecewise weight kept for comparison.
    """
    mode = candidate.get("blendMode", "linear")
    if mode == "nearest":
        return 0.0 if t < 0.5 else 1.0
    if mode == "smoothstep":
        return t * t * (3 - 2 * t)
    if mode == "linear":
        return t
    start = float(candidate.get("blendStart", 0.0))
    end = float(candidate.get("blendEnd", 1.0))
    if start == 0 and end == 1:
        return t
    u = float(np.clip((t - start) / (end - start), 0, 1))
    return u * u * (3 - 2 * u)


def smoothstep(low, high, value):
    t = np.clip((value - low) / (high - low), 0, 1)
    return t * t * (3 - 2 * t)


def gradient_energy(rgb):
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] +
            0.114 * rgb[..., 2])
    return float(np.mean(np.abs(gray[:, 1:] - gray[:, :-1])) +
                 np.mean(np.abs(gray[1:, :] - gray[:-1, :])))


def encode_key_atlas(full: dict[int, np.ndarray], keys: np.ndarray,
                     width: int, height: int, quality: int):
    rows = math.ceil(len(keys) / KEY_COLS)
    atlas = np.zeros((rows * height, KEY_COLS * width, 3), np.uint8)
    for slot, frame in enumerate(keys):
        x, y = slot % KEY_COLS * width, slot // KEY_COLS * height
        atlas[y:y + height, x:x + width] = full[int(frame)]
    buffer = io.BytesIO()
    Image.fromarray(atlas).save(buffer, "WEBP", quality=quality, method=4)
    data = buffer.getvalue()
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    image.load()
    return data, np.asarray(image), rows


def quantized_flow(current: np.ndarray, target: np.ndarray,
                   full_size: tuple[int, int], flow_scale: float,
                   smooth: bool):
    a = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    # Farneback flow is in flow-cell pixels. Encode full-canvas-pixel
    # displacement; the shader adds it directly to uv * u_resolution.
    flow = cv2.calcOpticalFlowFarneback(a, b, None,
                                       0.5, 3, 15, 3, 5, 1.2, 0)
    flow[..., 0] *= full_size[0] / current.shape[1]
    flow[..., 1] *= full_size[1] / current.shape[0]
    if smooth:
        flow = cv2.GaussianBlur(flow, (3, 3), 0)
    qx = np.clip(np.rint(flow[..., 0] / flow_scale + FLOW_OFFSET), 0, 255)
    qy = np.clip(np.rint(flow[..., 1] / flow_scale + FLOW_OFFSET), 0, 255)
    saturated = np.count_nonzero((qx <= 0) | (qx >= 255) |
                                 (qy <= 0) | (qy >= 255))
    return np.dstack([qx, qy,
                      np.full(qx.shape, FLOW_OFFSET)]).astype(np.uint8), saturated


def encode_flow_atlas(small: np.ndarray, keys: np.ndarray,
                      full_size: tuple[int, int], flow_scale: float,
                      smooth: bool, candidate: dict):
    n, cell_h, cell_w = small.shape[:3]
    flow_cells = []
    flow_cell0 = []
    flow_cell1 = []
    blend_weights = []
    saturated = 0
    total = 0
    for frame in range(n):
        lo, hi, t = brackets(keys, frame)
        weight = blend_weight(candidate, t)
        blend_weights.append(weight)

        def add_cell(key_slot):
            nonlocal saturated, total
            key_frame = int(keys[key_slot])
            if key_frame == frame:
                cell = np.full((cell_h, cell_w, 3), FLOW_OFFSET, np.uint8)
                count = 0
            else:
                cell, count = quantized_flow(
                    small[frame], small[key_frame], full_size,
                    flow_scale, smooth)
            saturated += count
            total += cell.shape[0] * cell.shape[1] * 2
            flow_cells.append(cell)
            return len(flow_cells) - 1

        if weight <= 1e-6:
            only = add_cell(lo)
            flow_cell0.append(only)
            flow_cell1.append(only)
        elif weight >= 1 - 1e-6:
            only = add_cell(hi)
            flow_cell0.append(only)
            flow_cell1.append(only)
        else:
            first = add_cell(lo)
            second = add_cell(hi)
            flow_cell0.append(first)
            flow_cell1.append(second)

    rows = math.ceil(len(flow_cells) / FLOW_COLS)
    atlas = np.full((rows * cell_h, FLOW_COLS * cell_w, 3), 128, np.uint8)
    for index, cell in enumerate(flow_cells):
        x = index % FLOW_COLS * cell_w
        y = index // FLOW_COLS * cell_h
        atlas[y:y + cell_h, x:x + cell_w] = cell
    buffer = io.BytesIO()
    if candidate.get("flowLossy"):
        # Decode the compressed atlas so every metric includes WebP's
        # actual effect on the packed motion values used by the shader.
        Image.fromarray(atlas).save(
            buffer, "WEBP", quality=int(candidate.get("flowQuality", 96)),
            method=6)
    else:
        Image.fromarray(atlas).save(buffer, "WEBP", lossless=True, method=6)
    data = buffer.getvalue()
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    image.load()
    return {
        "bytes": data,
        "atlas": np.asarray(image),
        "rows": rows,
        "saturation": saturated / total,
        "flowCell0": flow_cell0,
        "flowCell1": flow_cell1,
        "blendWeight": [round(w, 5) for w in blend_weights],
        "cells": len(flow_cells),
    }


def cell_at(flow: np.ndarray, index: int, cell_w: int, cell_h: int):
    x, y = index % FLOW_COLS * cell_w, index // FLOW_COLS * cell_h
    return flow[y:y + cell_h, x:x + cell_w].astype(np.float32)


def decode_flow(cell: np.ndarray, flow_scale: float,
                target_size: tuple[int, int], full_size: tuple[int, int]):
    flow = np.dstack([
        (cell[..., 0] - FLOW_OFFSET) * flow_scale,
        (cell[..., 1] - FLOW_OFFSET) * flow_scale,
    ])
    if (flow.shape[1], flow.shape[0]) != target_size:
        flow = cv2.resize(flow, target_size, interpolation=cv2.INTER_LINEAR)
    flow[..., 0] *= target_size[0] / full_size[0]
    flow[..., 1] *= target_size[1] / full_size[1]
    return flow


def key_image(atlas: np.ndarray, slot: int, width: int, height: int,
              small_size=None):
    x, y = slot % KEY_COLS * width, slot // KEY_COLS * height
    image = atlas[y:y + height, x:x + width]
    if small_size is not None:
        return cv2.resize(image, small_size, interpolation=cv2.INTER_AREA)
    return image


def warp_and_blend(key0, key1, flow0, flow1, weight):
    h, w = key0.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
    warp0 = cv2.remap(key0, grid_x + flow0[..., 0], grid_y + flow0[..., 1],
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    warp1 = cv2.remap(key1, grid_x + flow1[..., 0], grid_y + flow1[..., 1],
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return (warp0.astype(np.float32) * (1 - weight) +
            warp1.astype(np.float32) * weight)


def metrics(reference, output):
    error = np.abs(output.astype(np.float32) - reference.astype(np.float32))
    mse = np.mean((output.astype(np.float32) -
                   reference.astype(np.float32)) ** 2)
    return {
        "mae": round(float(error.mean()), 4),
        "psnr": round(float(10 * np.log10(255.0 ** 2 / max(mse, 1e-8))), 3),
        "sharpness": round(gradient_energy(np.clip(output, 0, 255)), 4),
    }


def evaluate_candidate(candidate, small, full, static, meta):
    n, cell_h, cell_w = small.shape[:3]
    width, height = meta["width"], meta["height"]
    keys = uniform_keys(n, candidate["step"])
    key_bytes, key_atlas, key_rows = encode_key_atlas(
        full, keys, width, height, candidate["quality"])
    flow = encode_flow_atlas(
        small, keys, (width, height), candidate["flowScale"],
        candidate["smoothFlow"], candidate)

    base_small = cv2.resize(static["base"], (cell_w, cell_h),
                            interpolation=cv2.INTER_AREA).astype(np.float32)
    alpha_small = cv2.resize(static["alpha"][..., 0], (cell_w, cell_h),
                             interpolation=cv2.INTER_AREA)[..., None]
    keys_small = [key_image(key_atlas, slot, width, height, (cell_w, cell_h))
                  for slot in range(len(keys))]
    small_outputs = []
    for frame in range(n):
        lo, hi, _ = brackets(keys, frame)
        weight = flow["blendWeight"][frame]
        flow0 = decode_flow(cell_at(flow["atlas"], flow["flowCell0"][frame],
                                    cell_w, cell_h),
                            candidate["flowScale"], (cell_w, cell_h),
                            (width, height))
        flow1 = decode_flow(cell_at(flow["atlas"], flow["flowCell1"][frame],
                                    cell_w, cell_h),
                            candidate["flowScale"], (cell_w, cell_h),
                            (width, height))
        warped = warp_and_blend(keys_small[lo], keys_small[hi],
                                flow0, flow1, weight)
        output = base_small * (1 - alpha_small) + warped * alpha_small
        small_outputs.append(output.astype(np.uint8))

    out_delta = np.abs(np.diff(np.stack(small_outputs).astype(np.float32),
                               axis=0))
    ref_delta = np.abs(np.diff(small.astype(np.float32), axis=0))
    nearest = np.abs(keys[None, :] - np.arange(n)[:, None]).argmin(1)
    switch_frames = [f for f in range(1, n) if nearest[f] != nearest[f - 1]]
    delta_energy = out_delta.mean((1, 2, 3))
    switch_delta = delta_energy[np.array(switch_frames) - 1]
    temporal = {
        "meanDeltaMae": round(float(np.abs(out_delta - ref_delta).mean()), 4),
        "meanOutputDelta": round(float(delta_energy.mean()), 4),
        "meanReferenceDelta": round(float(ref_delta.mean()), 4),
        "switchMeanDelta": round(float(switch_delta.mean()), 4),
        "switchSpikeRatio": round(float(switch_delta.mean() /
                                        np.median(delta_energy)), 4),
        "maxDelta": round(float(delta_energy.max()), 4),
        "switchFrames": switch_frames,
        "meanBlendWeight": round(float(np.mean(flow["blendWeight"])), 4),
    }

    samples = np.unique(np.linspace(0, n - 1, 15).round().astype(int))
    sample_results = []
    for frame in samples:
        lo, hi, t = brackets(keys, int(frame))
        weight = flow["blendWeight"][int(frame)]
        key0 = key_image(key_atlas, lo, width, height)
        key1 = key_image(key_atlas, hi, width, height)
        flow0 = decode_flow(cell_at(flow["atlas"],
                                    flow["flowCell0"][int(frame)],
                                    cell_w, cell_h),
                            candidate["flowScale"], (width, height),
                            (width, height))
        flow1 = decode_flow(cell_at(flow["atlas"],
                                    flow["flowCell1"][int(frame)],
                                    cell_w, cell_h),
                            candidate["flowScale"], (width, height),
                            (width, height))
        warped = warp_and_blend(key0, key1, flow0, flow1, weight)
        output = static["base"] * (1 - static["alpha"]) + warped * static["alpha"]
        sample_results.append({
            "frame": int(frame),
            "keyframes": [int(keys[lo]), int(keys[hi])],
            "mix": round(weight, 4),
            **metrics(full[int(frame)], output),
        })

    timeline_copy = dict(meta)
    timeline_copy["referenceFrames"] = {
        "keyframes": keys.tolist(),
        "flowCells": flow["cells"],
        "flowCell0": flow["flowCell0"],
        "flowCell1": flow["flowCell1"],
        "blendWeight": flow["blendWeight"],
        "flowCellWidth": cell_w,
        "flowCellHeight": cell_h,
        "flowScale": candidate["flowScale"],
        "blendStart": candidate.get("blendStart", 0.0),
        "blendEnd": candidate.get("blendEnd", 1.0),
        "flowOffset": FLOW_OFFSET,
    }
    timeline_gz = len(gzip.compress(
        json.dumps(timeline_copy, separators=(",", ":")).encode(),
        compresslevel=9, mtime=0))
    package_bytes = (len(key_bytes) + len(flow["bytes"]) + timeline_gz +
                     static["assetBytes"]["base"] +
                     static["assetBytes"]["mask"])
    return {
        "candidate": candidate,
        "keyframeCount": len(keys),
        "keyAtlasBytes": len(key_bytes),
        "flowAtlasBytes": len(flow["bytes"]),
        "flowCells": flow["cells"],
        "timelineGzipBytes": timeline_gz,
        "packageBytes": package_bytes,
        "flowSaturatedChannelRatio": round(flow["saturation"], 6),
        "flowAtlas": [FLOW_COLS * cell_w, flow["rows"] * cell_h],
        "meanMae": round(float(np.mean([s["mae"] for s in sample_results])), 4),
        "meanPsnr": round(float(np.mean([s["psnr"] for s in sample_results])), 3),
        "meanSharpness": round(float(np.mean(
            [s["sharpness"] for s in sample_results])), 4),
        "temporal": temporal,
        "samples": sample_results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    ap.add_argument("--sweep", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sweep = json.loads(args.sweep.read_text(encoding="utf-8"))
    meta = json.loads((ASSETS / "timeline.json").read_text(encoding="utf-8"))
    n = meta["frameCount"]
    samples = set(np.unique(np.linspace(0, n - 1, 15).round().astype(int)).tolist())
    union_keys = set()
    for candidate in sweep:
        union_keys.update(uniform_keys(n, candidate["step"]).tolist())
    cell_sizes = {tuple(c["flowCell"]) for c in sweep}
    if len(cell_sizes) > 1:
        raise ValueError("All sweep candidates must use the same flow cell")
    small, full, count = read_video(args.video, samples | union_keys,
                                    cell_sizes.pop())
    if count != n:
        raise ValueError(f"Expected {n} frames, decoded {count}")

    base = np.asarray(Image.open(ASSETS / "base.webp").convert("RGB"),
                      np.float32)
    mask = np.asarray(Image.open(ASSETS / "motion-mask.webp").convert("L"),
                      np.float32) / 255.0
    alpha = smoothstep(0.055, 0.145, mask)[..., None].astype(np.float32)
    static = {
        "base": base,
        "alpha": alpha,
        "assetBytes": {
            "base": (ASSETS / "base.webp").stat().st_size,
            "mask": (ASSETS / "motion-mask.webp").stat().st_size,
        },
    }
    results = []
    for candidate in sweep:
        result = evaluate_candidate(candidate, small, full, static, meta)
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "samples"},
                         indent=2))
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
