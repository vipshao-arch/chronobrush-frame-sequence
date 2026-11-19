import cv2
import numpy as np


path = r"C:\Users\vipshao\Desktop\参考 (2).mp4"
cap = cv2.VideoCapture(path)
ok, first = cap.read()
assert ok
lo = first.copy()
hi = first.copy()
sum_abs = np.zeros(first.shape[:2], np.float32)
count = 1
while True:
    ok, frame = cap.read()
    if not ok:
        break
    lo = np.minimum(lo, frame)
    hi = np.maximum(hi, frame)
    sum_abs += np.abs(frame.astype(np.float32) - first.astype(np.float32)).mean(2)
    count += 1
cap.release()
mean_abs = sum_abs / (count - 1)
rng = (hi.astype(np.float32) - lo.astype(np.float32)).mean(2)
print("frames", count, "shape", first.shape)
for name, arr in (("range", rng), ("meanabs", mean_abs)):
    print(name, "p50", np.percentile(arr, 50), "p75", np.percentile(arr, 75),
          "p90", np.percentile(arr, 90), "p95", np.percentile(arr, 95),
          "p99", np.percentile(arr, 99), "max", arr.max())
for tr, mt in ((8, 2), (10, 3), (12, 4), (16, 5), (20, 6)):
    raw = ((rng > tr) | (mean_abs > mt)).astype(np.uint8)
    print("thresh", tr, mt, "dynamic", raw.mean())
    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    dil = cv2.dilate(closed, np.ones((5, 5), np.uint8))
    print("  after morphology", dil.mean())
    recon = first.astype(np.float32) * (1 - dil[..., None]) + np.zeros_like(first, np.float32)
    # Static-pixel drift if this were the locked mask.
    static = (1 - dil).astype(bool)
    print("  locked static drift", mean_abs[static].mean(), "dynamic coverage", dil.mean())

cv2.imwrite("analysis/sheet/range_native.png", (np.clip(rng * 4, 0, 255)).astype(np.uint8))
cv2.imwrite("analysis/sheet/meanabs_native.png", (np.clip(mean_abs * 10, 0, 255)).astype(np.uint8))
