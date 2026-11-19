import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import test from "node:test";

const timeline = JSON.parse(await readFile(
  new URL("../public/assets/timeline.json", import.meta.url)));

test("sparse full-resolution reference data covers every source frame", () => {
  assert.equal(timeline.frameCount, 145);
  assert.equal(timeline.fps, 24);
  assert.equal(timeline.keyframes.length, 145);
  assert.equal(timeline.keyframes.at(-1), 144);
  assert.ok(timeline.base.locked);
  assert.ok(timeline.base.staticPixelRatio > 0.25);

  const rf = timeline.referenceFrames;
  assert.equal(rf.frames, timeline.frameCount);
  assert.equal(rf.fullResolution, true);
  assert.equal(rf.keyframeFile, "keyframes.webp");
  assert.equal(rf.flowFile, "flow.webp");
  assert.equal(rf.keyframes.length, 19);
  assert.deepEqual(rf.keyframes.slice(0, 3), [0, 8, 16]);
  assert.equal(rf.keyframes.at(-1), 144);
  assert.equal(rf.blendMode, "linear");
  assert.equal(rf.flowCells, 271);
  assert.equal(rf.flowCell0.length, timeline.frameCount);
  assert.equal(rf.flowCell1.length, timeline.frameCount);
  assert.equal(rf.blendWeight.length, timeline.frameCount);
  assert.ok(rf.flowCell0.every((cell) =>
    Number.isInteger(cell) && cell >= 0 && cell < rf.flowCells));
  assert.ok(rf.flowCell1.every((cell) =>
    Number.isInteger(cell) && cell >= 0 && cell < rf.flowCells));
  assert.ok(rf.blendWeight.every((weight) => weight >= 0 && weight <= 1));
  assert.equal(rf.blendWeight[0], 0);
  assert.equal(rf.blendWeight.at(-1), 1);
  assert.equal(rf.keyframeWidth, timeline.width);
  assert.equal(rf.keyframeHeight, timeline.height);
  assert.ok(rf.keyframeCols * rf.keyframeRows >= rf.keyframes.length);
  assert.equal(rf.flowCols, 13);
  assert.equal(rf.flowRows, 21);
  assert.ok(rf.flowCols * rf.flowRows >= rf.flowCells);
  assert.equal(rf.flowCellWidth, 368);
  assert.equal(rf.flowCellHeight, 158);
  assert.equal(rf.flowScale, 0.5);
  assert.equal(rf.flowOffset, 128);
  assert.equal(rf.anchorKeyframe, undefined);
  assert.equal(timeline.referenceVideo, undefined);
  assert.equal(timeline.sheet, undefined);
});

test("runtime assets contain no MP4 or legacy low-resolution frame sheet", async () => {
  const names = await readdir(new URL("../public/assets/", import.meta.url));
  assert.ok(!names.some((name) => name.endsWith(".mp4")));
  assert.ok(!names.includes("reference.webp"));
  for (const name of ["base.webp", "motion-mask.webp", "timeline.json",
                      "keyframes.webp", "flow.webp"]) {
    assert.ok(names.includes(name), `${name} is missing`);
  }
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
  assert.equal(timeline.motionSeries.displacement.length,
               timeline.keyframes.length - 1);
  assert.equal(timeline.motionSeries.deformation.length,
               timeline.keyframes.length - 1);
  assert.ok(timeline.motionSeries.displacement.every(Number.isFinite));
});
