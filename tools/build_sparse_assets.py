"""Build a compact, full-resolution sparse-frame reconstruction dataset.

The source video is read only here. Runtime assets contain full-resolution
WebP keyframes, compact optical-flow maps, a locked base, a motion mask and
the existing drawing-command timeline. No MP4 or video decoder is shipped.
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


def keyframes(count: int, step: int):
    keys = list(range(0, count, step))
    if keys[-1] != count - 1:
        keys.append(count - 1)
    return np.array(keys, dtype=int)


def nearest_anchor_indices(keys: np.ndarray, count: int):
    distance = np.abs(keys[None, :] - np.arange(count)[:, None])
    return distance.argmin(1).astype(int)


def make_key_atlas(frames: np.ndarray, keys: np.ndarray, width: int,
                   height: int):
    rows = math.ceil(len(keys) / KEY_COLS)
    sheet = np.zeros((rows * height, KEY_COLS * width, 3), np.uint8)
    for slot, index in enumerate(keys):
        x, y = (slot % KEY_COLS) * width, (slot // KEY_COLS) * height
        sheet[y:y + height, x:x + width] = frames[index]
    return sheet, rows


def make_flow_atlas(frames: np.ndarray, keys: np.ndarray,
                    anchors: np.ndarray, width: int, height: int):
    grays = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    cell_w, cell_h = FLOW_CELL
    rows = math.ceil(len(frames) / FLOW_COLS)
    sheet = np.full((rows * cell_h, FLOW_COLS * cell_w, 3), 128, np.uint8)
    scale_x = width / cell_w
    scale_y = height / cell_h
    for frame_index in range(len(frames)):
        anchor_slot = int(anchors[frame_index])
        anchor_frame = int(keys[anchor_slot])
        if anchor_frame == frame_index:
            flow = np.zeros((cell_h, cell_w, 2), np.float32)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                grays[frame_index], grays[anchor_frame], None,
                0.5, 3, 15, 3, 5, 1.2, 0)
            flow[..., 0] *= scale_x
            flow[..., 1] *= scale_y
        qx = np.clip(np.rint(flow[..., 0] / FLOW_SCALE + 128), 0, 255)
        qy = np.clip(np.rint(flow[..., 1] / FLOW_SCALE + 128), 0, 255)
        cell = np.dstack([qx.astype(np.uint8), qy.astype(np.uint8),
                          np.full(qx.shape, 128, np.uint8)])
        x = (frame_index % FLOW_COLS) * cell_w
        y = (frame_index // FLOW_COLS) * cell_h
        sheet[y:y + cell_h, x:x + cell_w] = cell
    return sheet, rows, cell_w, cell_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=SOURCE_ASSETS)
    ap.add_argument("--step", type=int, default=KEYFRAME_STEP)
    ap.add_argument("--quality", type=int, default=98)
    args = ap.parse_args()

    meta = json.loads((SOURCE_ASSETS / "timeline.json").read_text(encoding="utf-8"))
    width, height = meta["width"], meta["height"]
    frames, small = read_video(args.video, width, height)
    if len(frames) != meta["frameCount"]:
        raise ValueError("Video frame count does not match timeline")

    keys = keyframes(len(frames), args.step)
    anchors = nearest_anchor_indices(keys, len(frames))
    key_sheet, key_rows = make_key_atlas(frames, keys, width, height)
    flow_sheet, flow_rows, flow_w, flow_h = make_flow_atlas(
        small, keys, anchors, width, height)

    save_webp(key_sheet, args.out / "keyframes.webp",
              quality=args.quality, method=4)
    save_webp(flow_sheet, args.out / "flow.webp", lossless=True, method=6)

    meta["referenceFrames"] = {
        "sync": "timeline.frameAtTime",
        "frames": len(frames),
        "source": "pre-analyzed sparse WebP keyframes and optical flow",
        "strategy": "full-resolution sparse keyframes + quantized optical flow",
        "keyframeFile": "keyframes.webp",
        "flowFile": "flow.webp",
        "keyframes": keys.tolist(),
        "anchorKeyframe": anchors.tolist(),
        "keyframeCols": KEY_COLS,
        "keyframeRows": key_rows,
        "keyframeWidth": width,
        "keyframeHeight": height,
        "flowCols": FLOW_COLS,
        "flowRows": flow_rows,
        "flowCellWidth": flow_w,
        "flowCellHeight": flow_h,
        "flowScale": FLOW_SCALE,
        "flowOffset": 128,
        "fullResolution": True,
    }
    meta.pop("sheet", None)
    (args.out / "timeline.json").write_text(
        json.dumps(meta, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({
        "frames": len(frames),
        "keyframes": len(keys),
        "keyAtlas": [key_sheet.shape[1], key_sheet.shape[0]],
        "flowAtlas": [flow_sheet.shape[1], flow_sheet.shape[0]],
        "assetBytes": sum((args.out / name).stat().st_size for name in
                          ["base.webp", "motion-mask.webp",
                           "keyframes.webp", "flow.webp", "timeline.json"]),
    }, indent=2))


if __name__ == "__main__":
    main()
