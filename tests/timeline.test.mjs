import assert from "node:assert/strict";
import test from "node:test";

import {
  findStateInterval,
  frameAtTime,
  interpolateState,
  keyframeForFrame,
  meanAbsError,
  sheetCell,
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

test("reference frames map to the correct sprite sheet cell", () => {
  const meta = {
    keyframes: [0, 3, 6, 9, 12],
    sheet: { cols: 2, cellWidth: 490, cellHeight: 210 },
  };
  assert.equal(keyframeForFrame(meta, 0), 0);
  assert.equal(keyframeForFrame(meta, 4), 1);
  const cell = sheetCell(meta, 10);
  assert.deepEqual([cell.index, cell.col, cell.row], [3, 1, 1]);
  assert.deepEqual([cell.sx, cell.sy], [490, 210]);
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
