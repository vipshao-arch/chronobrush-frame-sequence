"""Measure blended-keyframe reconstruction against the source video.

This mirrors the WebGL shader: each pair of full-resolution keyframes is
warped with its own decoded flow cell, blended with the timeline's continuous
weight, then combined with the immutable frame-0 base. The source MP4 is read
only by this offline check. Temporal metrics use all frames at flow-cell
resolution; pixel metrics use evenly sampled full-resolution frames.
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


def atlas_cell(atlas, index, cols, cell_w, cell_h):
    x = index % cols * cell_w
    y = index // cols * cell_h
    return np.asarray(atlas.crop((x, y, x + cell_w, y + cell_h)), np.float32)


def decoded_flow(atlas, index, rf, target_size, full_size):
    cell_w, cell_h = rf["flowCellWidth"], rf["flowCellHeight"]
    packed = atlas_cell(atlas, index, rf["flowCols"], cell_w, cell_h)
    dx = (packed[..., 0] - rf["flowOffset"]) * rf["flowScale"]
    dy = (packed[..., 1] - rf["flowOffset"]) * rf["flowScale"]
    flow = np.dstack([dx, dy])
    if target_size != (cell_w, cell_h):
        flow = cv2.resize(flow, target_size, interpolation=cv2.INTER_LINEAR)
    flow[..., 0] *= target_size[0] / full_size[0]
    flow[..., 1] *= target_size[1] / full_size[1]
    return flow


def key_image(atlas, slot, width, height, cols, target_size=None):
    x = slot % cols * width
    y = slot // cols * height
    image = np.asarray(atlas.crop((x, y, x + width, y + height)))
    if target_size is not None:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return image


def brackets(keys, frame):
    lo = int(np.clip(np.searchsorted(keys, frame, side="right") - 1,
                     0, len(keys) - 2))
    return lo, lo + 1


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
    full_size = (width, height)
    flow_size = (rf["flowCellWidth"], rf["flowCellHeight"])
    keys = np.asarray(rf["keyframes"], dtype=int)
    weights = rf["blendWeight"]
    base = np.asarray(Image.open(args.assets / meta["base"]["file"])
                      .convert("RGB"), np.float32)
    mask = np.asarray(Image.open(args.assets / meta["mask"]["file"])
                      .convert("L"), np.float32) / 255.0
    alpha = smoothstep(0.055, 0.145, mask)[..., None].astype(np.float32)
    alpha_small = cv2.resize(alpha[..., 0], flow_size,
                             interpolation=cv2.INTER_AREA)[..., None]
    base_small = cv2.resize(base, flow_size,
                            interpolation=cv2.INTER_AREA).astype(np.float32)
    key_atlas = Image.open(args.assets / rf["keyframeFile"]).convert("RGB")
    flow_atlas = Image.open(args.assets / rf["flowFile"]).convert("RGB")
    key_atlas.load()
    flow_atlas.load()
    locked = alpha[..., 0] == 0
    indices = np.unique(np.linspace(
        0, meta["frameCount"] - 1, min(args.samples, meta["frameCount"]))
        .round().astype(int))
    wanted = set(indices.tolist())

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    references_small = []
    outputs_small = []
    first_locked = None
    results = []
    for frame_number in range(meta["frameCount"]):
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Video ended early at frame {frame_number}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        reference_small = cv2.resize(rgb, flow_size,
                                     interpolation=cv2.INTER_AREA)
        references_small.append(reference_small)
        lo, hi = brackets(keys, frame_number)
        weight = weights[frame_number]
        key0 = key_image(key_atlas, lo, width, height,
                         rf["keyframeCols"], flow_size)
        key1 = key_image(key_atlas, hi, width, height,
                         rf["keyframeCols"], flow_size)
        flow0 = decoded_flow(flow_atlas, rf["flowCell0"][frame_number],
                             rf, flow_size, full_size)
        flow1 = decoded_flow(flow_atlas, rf["flowCell1"][frame_number],
                             rf, flow_size, full_size)
        warped_small = warp_and_blend(key0, key1, flow0, flow1, weight)
        output_small = (base_small * (1 - alpha_small) +
                        warped_small * alpha_small)
        outputs_small.append(np.clip(output_small, 0, 255).astype(np.uint8))

        if frame_number in wanted:
            key0_full = key_image(key_atlas, lo, width, height,
                                  rf["keyframeCols"])
            key1_full = key_image(key_atlas, hi, width, height,
                                  rf["keyframeCols"])
            flow0_full = decoded_flow(flow_atlas,
                                      rf["flowCell0"][frame_number],
                                      rf, full_size, full_size)
            flow1_full = decoded_flow(flow_atlas,
                                      rf["flowCell1"][frame_number],
                                      rf, full_size, full_size)
            warped = warp_and_blend(key0_full, key1_full,
                                    flow0_full, flow1_full, weight)
            if first_locked is None:
                first_locked = warped[locked]
            output = base * (1 - alpha) + warped * alpha
            reference = rgb.astype(np.float32)
            error = np.abs(output - reference)
            mse = np.mean((output - reference) ** 2)
            results.append({
                "frame": frame_number,
                "time": round(frame_number / meta["fps"], 6),
                "keyFrame0": int(keys[lo]),
                "keyFrame1": int(keys[hi]),
                "mix": weight,
                "mae": round(float(error.mean()), 5),
                "p95": round(float(np.percentile(error, 95)), 5),
                "psnr": round(float(10 * np.log10(
                    255.0 ** 2 / max(mse, 1e-8))), 3),
                "sharpness": round(gradient_energy(
                    np.clip(output, 0, 255)), 4),
                "referenceSharpness": round(gradient_energy(reference), 4),
                "lockedReferenceDrift": round(float(np.abs(
                    warped[locked] - first_locked).mean()), 5),
            })
    cap.release()

    ref_delta = np.abs(np.diff(np.stack(references_small).astype(np.float32),
                               axis=0))
    out_delta = np.abs(np.diff(np.stack(outputs_small).astype(np.float32),
                               axis=0))
    delta_energy = out_delta.mean((1, 2, 3))
    nearest = np.abs(keys[None, :] -
                     np.arange(meta["frameCount"])[:, None]).argmin(1)
    switch_frames = [f for f in range(1, meta["frameCount"])
                     if nearest[f] != nearest[f - 1]]
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
        "meanBlendWeight": round(float(np.mean(weights)), 4),
    }
    summary = {
        "sourceFrames": meta["frameCount"],
        "fps": meta["fps"],
        "sampledFrames": len(results),
        "width": width,
        "height": height,
        "strategy": rf["strategy"],
        "keyframes": len(rf["keyframes"]),
        "flowCells": rf["flowCells"],
        "blendMode": rf["blendMode"],
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
        "temporal": temporal,
        "baseTextureUpdates": 0,
        "frames": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "frames"},
                     indent=2))


if __name__ == "__main__":
    main()
