import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const timeline = JSON.parse(await readFile(new URL("../public/assets/timeline.json", import.meta.url)));

test("pre-analysis timeline covers every source frame", () => {
  assert.equal(timeline.frameCount, 145);
  assert.equal(timeline.fps, 24);
  assert.equal(timeline.keyframes.length, 145);
  assert.equal(timeline.keyframes.at(-1), 144);
  assert.equal(timeline.sheet.cols * timeline.sheet.rows, 147);
  assert.ok(timeline.sheet.cols * timeline.sheet.rows >= timeline.keyframes.length);
  assert.ok(timeline.base.locked);
  assert.ok(timeline.base.staticPixelRatio > 0.25);
  assert.equal(timeline.referenceFrames.frames, timeline.keyframes.length);
  assert.equal(timeline.referenceFrames.file, "reference.webp");
  assert.equal(timeline.referenceVideo, undefined);
});

test("each drawing command has synchronized time-varying state", () => {
  assert.equal(timeline.commands.length, 96);
  for (const command of timeline.commands) {
    assert.ok(["ellipse", "line", "bezier", "brush"].includes(command.type));
    assert.equal(command.points.length, 4);
    assert.equal(command.states.length, timeline.keyframes.length);
    assert.equal(command.motion.length, timeline.keyframes.length - 1);
    for (const state of command.states) {
      assert.equal(state.xy.length, 2);
      assert.equal(state.shape.length, 2);
      assert.equal(state.color.length, 3);
      assert.ok(state.alpha >= 0 && state.alpha <= 1);
    }
  }
});

test("motion series matches every keyframe transition", () => {
  assert.equal(timeline.motionSeries.displacement.length, timeline.keyframes.length - 1);
  assert.equal(timeline.motionSeries.deformation.length, timeline.keyframes.length - 1);
  assert.ok(timeline.motionSeries.displacement.every(Number.isFinite));
});
