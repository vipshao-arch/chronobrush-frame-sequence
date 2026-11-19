import assert from "node:assert/strict";
import test from "node:test";

import {
  findStateInterval,
  frameAtTime,
  interpolateState,
  meanAbsError,
  sparseFrameInfo,
  transformPoints,
} from "../src/timeline.js";

test("video time maps to the exact 24 fps source frame", () => {
  assert.equal(frameAtTime(0, 24, 145), 0);
  assert.equal(frameAtTime(1 / 24, 24, 145), 1);
  assert.equal(frameAtTime(6.03, 24, 145), 144);
  assert.equal(frameAtTime(-1, 24, 145), 0);
});

test("state interpolation preserves position and deformation over time", () => {
  const states = [
    { t: 0, frame: 0, xy: [0, 0], rotation: 0, shape: [1, 1], color: [10, 20, 30], alpha: 0.2 },
    { t: 1, frame: 24, xy: [10, -4], rotation: 0.2, shape: [1.4, 0.8], color: [30, 40, 50], alpha: 0.4 },
  ];
  assert.equal(findStateInterval(states, 1.5)[1], states[1]);
  const middle = interpolateState(states, 0.5);
  assert.deepEqual(middle.xy, [5, -2]);
  assert.equal(middle.rotation, 0.1);
  assert.deepEqual(middle.shape, [1.2, 0.9]);
  assert.deepEqual(middle.color, [20, 30, 40]);
  assert.ok(Math.abs(middle.alpha - 0.3) < 1e-12);
});

test("sparse reference frames map to a continuous keyframe bracket", () => {
  const meta = {
    referenceFrames: {
      keyframes: [0, 3, 6, 9, 12],
      keyframeCols: 2,
      flowCols: 4,
      flowCell0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
      flowCell1: [0, 2, 4, 3, 6, 8, 6, 10, 12, 9, 14, 16, 12],
      blendWeight: [0, 1/3, 2/3, 0, 1/3, 2/3, 0, 1/3, 2/3, 0, 1/3, 2/3, 1],
    },
  };
  const first = sparseFrameInfo(meta, 0);
  assert.deepEqual([first.keyFrame0, first.keyFrame1, first.mix],
                   [0, 3, 0]);
  assert.deepEqual(first.keyCell0, [0, 0]);
  assert.deepEqual(first.keyCell1, [1, 0]);
  assert.deepEqual(first.flowCell0, [0, 0]);
  assert.deepEqual(first.flowCell1, [0, 0]);
  const middle = sparseFrameInfo(meta, 5);
  assert.deepEqual([middle.keyFrame0, middle.keyFrame1, middle.mix],
                   [3, 6, 2/3]);
  assert.deepEqual(middle.keyCell0, [1, 0]);
  assert.deepEqual(middle.keyCell1, [0, 1]);
  assert.deepEqual(middle.flowCell0, [1, 1]);
  assert.deepEqual(middle.flowCell1, [0, 2]);
  const last = sparseFrameInfo(meta, 12);
  assert.deepEqual([last.keyFrame0, last.keyFrame1, last.mix],
                   [9, 12, 1]);
});

test("brush points undergo affine shape and displacement", () => {
  const state = { xy: [5, 7], rotation: 0, shape: [2, 1], color: [0, 0, 0], alpha: 1 };
  const points = transformPoints([[0, 0], [10, 0]], state);
  assert.deepEqual(points, [[0, 7], [20, 7]]);
});

test("pixel error is the mean absolute channel difference", () => {
  const a = new Uint8ClampedArray([0, 0, 0, 255, 10, 20, 30, 255]);
  const b = new Uint8ClampedArray([3, 6, 9, 255, 10, 20, 30, 255]);
  assert.equal(meanAbsError(a, b), 3);
});
