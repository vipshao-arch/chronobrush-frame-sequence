export function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export function frameAtTime(time, fps, frameCount) {
  return clamp(Math.round(time * fps), 0, frameCount - 1);
}

export function findStateInterval(states, time) {
  if (!states.length) return null;
  let lo = 0;
  let hi = states.length - 1;
  if (time <= states[0].t) return [states[0], states[0], 0];
  if (time >= states[hi].t) return [states[hi], states[hi], 0];
  while (lo + 1 < hi) {
    const mid = (lo + hi) >> 1;
    if (states[mid].t <= time) lo = mid;
    else hi = mid;
  }
  const span = states[hi].t - states[lo].t;
  return [states[lo], states[hi], span > 0 ? (time - states[lo].t) / span : 0];
}

const mix = (a, b, t) => a + (b - a) * t;

export function interpolateState(states, time) {
  const interval = findStateInterval(states, time);
  if (!interval) return null;
  const [a, b, u] = interval;
  return {
    t: time,
    frame: u < 0.5 ? a.frame : b.frame,
    xy: [mix(a.xy[0], b.xy[0], u), mix(a.xy[1], b.xy[1], u)],
    rotation: mix(a.rotation, b.rotation, u),
    shape: [mix(a.shape[0], b.shape[0], u), mix(a.shape[1], b.shape[1], u)],
    color: a.color.map((v, i) => Math.round(mix(v, b.color[i], u))),
    alpha: mix(a.alpha, b.alpha, u),
  };
}

export function transformPoints(points, state) {
  if (!points.length) return [];
  const cx = points.reduce((sum, p) => sum + p[0], 0) / points.length;
  const cy = points.reduce((sum, p) => sum + p[1], 0) / points.length;
  const cos = Math.cos(state.rotation);
  const sin = Math.sin(state.rotation);
  return points.map(([x, y]) => {
    const sx = (x - cx) * state.shape[0];
    const sy = (y - cy) * state.shape[1];
    return [cx + sx * cos - sy * sin + state.xy[0],
            cy + sx * sin + sy * cos + state.xy[1]];
  });
}

export function sparseFrameInfo(meta, frame) {
  const rf = meta.referenceFrames;
  const keys = rf.keyframes;
  let key0 = keys.length - 2;
  for (let i = 0; i < keys.length - 1; i++) {
    if (frame >= keys[i] && frame <= keys[i + 1]) {
      key0 = i;
      break;
    }
  }
  const key1 = key0 + 1;
  const span = keys[key1] - keys[key0];
  const mix = clamp((frame - keys[key0]) / span, 0, 1);
  const cellIndex0 = rf.flowCell0[frame];
  const cellIndex1 = rf.flowCell1[frame];
  return {
    key0,
    key1,
    keyFrame0: keys[key0],
    keyFrame1: keys[key1],
    anchorDistance: Math.min(frame - keys[key0], keys[key1] - frame),
    mix: rf.blendWeight[frame],
    keyCell0: [key0 % rf.keyframeCols, Math.floor(key0 / rf.keyframeCols)],
    keyCell1: [key1 % rf.keyframeCols, Math.floor(key1 / rf.keyframeCols)],
    flowCell0: [cellIndex0 % rf.flowCols,
                Math.floor(cellIndex0 / rf.flowCols)],
    flowCell1: [cellIndex1 % rf.flowCols,
                Math.floor(cellIndex1 / rf.flowCols)],
  };
}

export function meanAbsError(a, b) {
  if (a.length !== b.length) throw new Error("Image buffers differ in size");
  let sum = 0;
  let count = 0;
  for (let i = 0; i < a.length; i += 4) {
    sum += Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) +
           Math.abs(a[i + 2] - b[i + 2]);
    count += 3;
  }
  return sum / count;
}
