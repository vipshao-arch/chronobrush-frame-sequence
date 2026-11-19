"""Extract a locked WebGL base and per-frame, reference-driven drawing commands.

Input: the reference MP4 (used only during pre-analysis). Output: a static
WebP, a WebP sprite sheet containing every reference frame, a WebP motion
mask, and a JSON timeline of animated primitives. The browser needs only
these generated data files; the source MP4 is never shipped at runtime.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read_frames(path: Path, width: int, height: int, fps: float):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    out = []
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not out:
        raise RuntimeError("No video frames found")
    return np.stack(out), source_fps or fps


def save_webp(rgb: np.ndarray, path: Path, quality: int = 92, method: int = 6,
              lossless: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, "WEBP", quality=quality, method=method,
                              lossless=lossless)


def motion_heatmap(frames: np.ndarray, sample: int = 3) -> np.ndarray:
    f = frames[::sample].astype(np.float32)
    temporal = np.median(f, axis=0)
    diff = np.mean(np.abs(f - temporal), axis=(0, 3))
    spread = (f.max(0) - f.min(0)).mean(2)
    # Two independent statistics: persistent change and large excursions.
    heat = np.maximum(diff / 14.0, spread / 55.0)
    return np.clip(heat, 0.0, 1.0).astype(np.float32)


def locked_mask(heat: np.ndarray) -> np.ndarray:
    # A conservative mask: only pixels that barely move are part of the
    # immutable WebGL base. Feathering lives in the shader, so this mask is hard.
    stable = (heat < 0.10).astype(np.uint8)
    stable = cv2.morphologyEx(stable, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    stable = cv2.morphologyEx(stable, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.GaussianBlur(stable.astype(np.float32), (0, 0), 1.4)


def median_base(frames: np.ndarray, static: np.ndarray) -> np.ndarray:
    # The first reference frame is the immutable ground truth for locked pixels.
    # The motion mask in the shader replaces all other pixels before display.
    return frames[0].copy()


def temporal_keyframes(frames: np.ndarray, count: int):
    n = len(frames)
    if count >= n:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, count).round().astype(int))


def make_sheet(frames: np.ndarray, indices: np.ndarray, cols: int,
               cell_width: int, cell_height: int) -> tuple[np.ndarray, int]:
    rows = math.ceil(len(indices) / cols)
    sheet = np.zeros((rows * cell_height, cols * cell_width, 3), np.uint8)
    for k, idx in enumerate(indices):
        cell = cv2.resize(frames[idx], (cell_width, cell_height),
                          interpolation=cv2.INTER_AREA)
        y, x = divmod(k, cols)
        sheet[y * cell_height:(y + 1) * cell_height,
              x * cell_width:(x + 1) * cell_width] = cell
    return sheet, rows


def sample_cubic(points: np.ndarray, t: float) -> tuple[float, float]:
    if len(points) == 0:
        return 0.0, 0.0
    x = np.clip(t * (len(points) - 1), 0, len(points) - 1)
    i = min(int(math.floor(x)), len(points) - 2)
    u = x - i if len(points) > 1 else 0.0
    p1 = points[i].astype(np.float32)
    p2 = points[min(i + 1, len(points) - 1)].astype(np.float32)
    p0 = points[max(0, i - 1)].astype(np.float32)
    p3 = points[min(i + 2, len(points) - 1)].astype(np.float32)
    q = 0.5 * ((2 * p1) + (-p0 + p2) * u +
               (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u +
               (-p0 + 3 * p1 - 3 * p2 + p3) * u * u * u)
    return float(q[0]), float(q[1])


def sample_color(rgb: np.ndarray, x: float, y: float) -> list[int]:
    h, w = rgb.shape[:2]
    xi = int(np.clip(round(x), 0, w - 1))
    yi = int(np.clip(round(y), 0, h - 1))
    patch = rgb[max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2]
    return np.median(patch.reshape(-1, 3), axis=0).astype(int).tolist()


def flow_sequence(frames: np.ndarray, keys: np.ndarray) -> list[np.ndarray]:
    """Optical flow at each keyframe transition, in full-resolution units."""
    flows = []
    prev = cv2.cvtColor(cv2.resize(frames[keys[0]], None, fx=0.5, fy=0.5,
                                   interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    for idx in keys[1:]:
        gray = cv2.cvtColor(cv2.resize(frames[idx], None, fx=0.5, fy=0.5,
                                       interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(flow * 2.0)
        prev = gray
    return flows


def stroke_motion(points: np.ndarray, flow: np.ndarray) -> list[float]:
    h, w = flow.shape[:2]
    q = points / 2.0
    q[:, 0] = np.clip(q[:, 0], 0, w - 1)
    q[:, 1] = np.clip(q[:, 1], 0, h - 1)
    xi = np.round(q[:, 0]).astype(int)
    yi = np.round(q[:, 1]).astype(int)
    v = flow[yi, xi]
    mean = v.mean(0)
    # Fit the residual local flow around the stroke centroid to a 2x2 affine
    # transform. Its antisymmetric part is rotation; its symmetric part is
    # stretch/shear (deformation).
    p = (q - q.mean(0)) / 20.0
    r = (v - mean) / 4.0
    affine = np.linalg.lstsq(p, r, rcond=None)[0].T
    rotation = math.atan2(affine[1, 0] - affine[0, 1],
                          2.0 + affine[0, 0] + affine[1, 1])
    symmetric = (affine + affine.T) / 2.0
    deformation = float(np.clip(np.linalg.norm(symmetric), 0, 1))
    return [round(float(mean[0]), 3), round(float(mean[1]), 3),
            round(float(np.clip(rotation, -0.25, 0.25)), 5),
            round(deformation, 5)]


def extract_commands(frames: np.ndarray, keys: np.ndarray, heat: np.ndarray,
                     threshold: float, primitives: int, flows: list[np.ndarray],
                     fps: float):
    h, w = frames.shape[1:3]
    # Sum motion along a modestly downsampled grid to find candidate contours.
    small = cv2.resize(heat, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    salience = cv2.GaussianBlur(small, (0, 0), 1.1)
    salience = np.clip((salience - threshold * 0.6) / max(0.08, 1 - threshold * 0.6), 0, 1)
    edges = cv2.Canny((salience * 255).astype(np.uint8), 45, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    selected = []
    rng = np.random.default_rng(1729)
    for contour in contours[:primitives * 3]:
        if cv2.contourArea(contour) < 8 and cv2.arcLength(contour, False) < 18:
            continue
        arc = cv2.approxPolyDP(contour, 2.5, True).reshape(-1, 2)
        if len(arc) < 4:
            continue
        # Several contour samples become independent bezier/line/ellipse strokes.
        for start in range(0, len(arc), max(3, len(arc) // 3)):
            pts = np.roll(arc, -start, axis=0)[:8] * 2.0
            if len(pts) < 4:
                continue
            mx, my = pts[:4].mean(0)
            if salience[int(np.clip(my / 2, 0, small.shape[0] - 1)),
                        int(np.clip(mx / 2, 0, small.shape[1] - 1))] < 0.04:
                continue
            selected.append(pts.astype(np.float32))
            if len(selected) >= primitives:
                break
        if len(selected) >= primitives:
            break
    # Fill missing slots from a coarse grid of strong motion points.
    ys, xs = np.where(cv2.resize(heat, (w // 16, h // 16)) > threshold)
    while len(selected) < primitives and len(xs):
        j = int(rng.integers(0, len(xs)))
        cx, cy = xs[j] * 16.0, ys[j] * 16.0
        angle = float(rng.uniform(0, math.tau))
        stretch = float(rng.uniform(10, 35))
        selected.append(np.array([
            [cx, cy], [cx + stretch * math.cos(angle), cy + stretch * math.sin(angle)],
            [cx + stretch * 1.7 * math.cos(angle), cy + stretch * 1.7 * math.sin(angle)],
            [cx + stretch * 2.4 * math.cos(angle), cy + stretch * 2.4 * math.sin(angle)],
        ], np.float32))

    commands = []
    motions = [[stroke_motion(points, flow) for flow in flows]
               for points in selected[:primitives]]
    commands = []
    for pid, points in enumerate(selected[:primitives]):
        kind = ("bezier", "ellipse", "line", "brush")[pid % 4]
        colors = []
        for idx in keys:
            x, y = sample_cubic(points, 0.5)
            colors.append(sample_color(frames[idx], x, y))
        states = []
        tx = ty = rotation = stretch = 0.0
        for k in range(len(keys)):
            if k > 0:
                m = motions[pid][k - 1]
                tx = float(np.clip(tx + m[0] * 0.8, -36, 36))
                ty = float(np.clip(ty + m[1] * 0.8, -36, 36))
                rotation = float(np.clip(rotation + m[2] * 0.6, -0.32, 0.32))
                stretch = float(np.clip(stretch + (m[3] - 0.5) * 0.018, -0.2, 0.2))
            states.append({
                "t": round(float(keys[k]) / fps, 6),
                "frame": int(keys[k]),
                "xy": [round(tx, 3), round(ty, 3)],
                "rotation": round(rotation, 5),
                "shape": [round(1.0 + stretch, 4), round(1.0 - stretch * 0.7, 4)],
                "color": colors[k],
                "alpha": round(float(np.clip(0.18 + heat[
                    int(np.clip(points[:, 1].mean(), 0, h - 1)),
                    int(np.clip(points[:, 0].mean(), 0, w - 1))] * 0.42, 0.18, 0.58)), 4),
            })
        commands.append({
            "id": f"stroke-{pid:04d}",
            "type": kind,
            "points": [[round(float(x), 2), round(float(y), 2)] for x, y in points[:4]],
            "width": round(float(np.clip(cv2.arcLength(points.astype(np.float32)[:4], False) / 14, 1.5, 12)), 2),
            "locked": False,
            "states": states,
            "motion": motions[pid],
        })
    return commands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("public/assets"))
    ap.add_argument("--width", type=int, default=1470)
    ap.add_argument("--height", type=int, default=630)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--keys", type=int, default=0,
                    help="0 means all source frames")
    ap.add_argument("--primitives", type=int, default=96)
    ap.add_argument("--threshold", type=float, default=0.18)
    args = ap.parse_args()

    frames, source_fps = read_frames(args.video, args.width, args.height, args.fps)
    heat = motion_heatmap(frames)
    static = locked_mask(heat)
    base = median_base(frames, static)
    key_count = args.keys or len(frames)
    keys = temporal_keyframes(frames, min(key_count, len(frames)))
    flows = flow_sequence(frames, keys)
    cols = 7
    cell_width, cell_height = 735, 315
    sheet, rows = make_sheet(frames, keys, cols, cell_width, cell_height)
    commands = extract_commands(frames, keys, heat, args.threshold,
                                args.primitives, flows, args.fps)

    save_webp(base, args.out / "base.webp", lossless=True, method=4)
    save_webp(sheet, args.out / "reference.webp", 91)
    mask = np.dstack([np.clip(heat * 255, 0, 255).astype(np.uint8)] * 3)
    save_webp(mask, args.out / "motion-mask.webp", lossless=True)
    stale_video = args.out / "reference.mp4"
    if stale_video.exists():
        stale_video.unlink()

    meta = {
        "version": 1,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "sourceFps": source_fps,
        "frameCount": len(frames),
        "duration": len(frames) / args.fps,
        "keyframes": keys.tolist(),
        "keyTimes": [round(float(i) / args.fps, 6) for i in keys],
        "referenceFrames": {
            "file": "reference.webp",
            "sync": "timeline.frameAtTime",
            "frames": len(keys),
            "source": "pre-analyzed WebP frames",
        },
        "sheet": {"file": "reference.webp", "cols": cols, "rows": rows,
                  "cellWidth": cell_width, "cellHeight": cell_height},
        "base": {"file": "base.webp", "locked": True, "source": "frame 0",
                 "staticPixelRatio": round(float((static > 0.5).mean()), 5)},
        "mask": {"file": "motion-mask.webp"},
        "commands": commands,
        "motionSeries": {
            "displacement": [round(float(np.mean([np.hypot(m[0], m[1]) for m in col])), 4)
                             for col in zip(*(c["motion"] for c in commands))] if commands else [],
            "deformation": [round(float(np.mean([m[3] for m in col])), 5)
                            for col in zip(*(c["motion"] for c in commands))] if commands else [],
        },
        "displacementStats": {
            "mean": round(float(np.mean(np.abs(frames[::6, 1:] - frames[::6, :-1]))), 4),
            "p95": round(float(np.percentile(np.abs(frames[::6, 1:] - frames[::6, :-1]), 95)), 4),
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "timeline.json").write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({
        "frames": len(frames), "keys": len(keys), "commands": len(commands),
        "staticPixelRatio": meta["base"]["staticPixelRatio"],
        "sheet": [sheet.shape[1], sheet.shape[0]],
        "assetBytes": sum((args.out / p).stat().st_size for p in
                          ["base.webp", "reference.webp", "motion-mask.webp",
                           "timeline.json"]),
    }, indent=2))


if __name__ == "__main__":
    main()
