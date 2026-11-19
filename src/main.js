import { renderCommands } from "./draw.js";
import { assetUrl } from "./assets.js";
import { frameAtTime, meanAbsError, sheetCell } from "./timeline.js";
import { loadRenderer } from "./webgl.js";

const $ = (id) => document.getElementById(id);
const els = {
  status: $("status"), stage: $("stage"), gl: $("gl"), vector: $("vector"),
  reference: $("reference"), play: $("play"), seek: $("seek"),
  time: $("time"), stepBack: $("stepBack"), stepForward: $("stepForward"),
  vectorToggle: $("vectorToggle"), splitToggle: $("splitToggle"),
  frameLabel: $("frameLabel"), frameValue: $("frameValue"), mae: $("mae"),
  locked: $("locked"), sync: $("sync"), displacement: $("displacement"),
  deformation: $("deformation"), motionTime: $("motionTime"), chart: $("chart"),
  cmdCount: $("cmdCount"), typeRow: $("typeRow"),
  commandList: $("commandList"), stageCaption: $("stageCaption"),
};

const state = {
  meta: null,
  renderer: null,
  sheetImage: null,
  playing: true,
  time: 0,
  lastFrame: -1,
  lastMeterAt: -1,
  lastLoopAt: 0,
};

function fitCanvas(canvas, width, height) {
  canvas.width = width;
  canvas.height = height;
}

function rounded(value, digits = 2) {
  return Number(value).toFixed(digits);
}

function renderCommandList(meta) {
  const counts = meta.commands.reduce((acc, c) => {
    acc[c.type] = (acc[c.type] || 0) + 1;
    return acc;
  }, {});
  els.cmdCount.textContent = `${meta.commands.length} 条 · ${meta.keyframes.length} 个采样时刻`;
  els.typeRow.innerHTML = ["ellipse", "line", "bezier", "brush"]
    .map((type) => `<span>${type}<b>${counts[type] || 0}</b></span>`).join("");
  els.commandList.innerHTML = meta.commands.slice(0, 12).map((c) => `
    <div class="command" data-command="${c.id}">
      <span class="id">${c.id}</span>
      <span class="type"><b>${c.type}</b> · ${rounded(c.width, 1)}px · ${c.points.length} 控制点</span>
      <span class="motion" id="motion-${c.id}">0.00px</span>
    </div>`).join("");
}

function drawChart(meta, time) {
  const ctx = els.chart.getContext("2d");
  const { width: w, height: h } = els.chart;
  const pad = 16;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#263141";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad + (h - pad * 2) * i / 4;
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(w - pad, y);
    ctx.stroke();
  }
  const series = meta.motionSeries;
  const n = series.displacement.length;
  if (!n) return;
  const maxD = Math.max(...series.displacement, 0.001);
  const maxS = Math.max(...series.deformation, 0.001);
  const x = (i) => pad + i / Math.max(1, n - 1) * (w - pad * 2);
  ctx.lineWidth = 1.6;
  for (const [values, max, color] of [
    [series.displacement, maxD, "#4cc9f0"],
    [series.deformation, maxS, "#f6bd60"],
  ]) {
    ctx.beginPath();
    values.forEach((v, i) => {
      const y = h - pad - v / max * (h - pad * 2);
      if (i === 0) ctx.moveTo(x(i), y);
      else ctx.lineTo(x(i), y);
    });
    ctx.strokeStyle = color;
    ctx.stroke();
  }
  const cursor = pad + Math.min(1, time / meta.duration) * (w - pad * 2);
  ctx.beginPath();
  ctx.moveTo(cursor, pad);
  ctx.lineTo(cursor, h - pad);
  ctx.strokeStyle = "#e85d75";
  ctx.stroke();
  els.motionTime.textContent = `t = ${rounded(time, 3)}s`;
}

function smallContext(width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  return ctx;
}

function measureQuality(meta, time, cell) {
  if (Math.abs(time - state.lastMeterAt) < 0.12 && state.lastFrame >= 0) return;
  state.lastMeterAt = time;
  const w = 147;
  const h = 63;
  const refCtx = smallContext(w, h);
  refCtx.drawImage(state.sheetImage, cell.sx, cell.sy,
    cell.cellWidth, cell.cellHeight, 0, 0, w, h);
  const reference = refCtx.getImageData(0, 0, w, h).data;

  // Read the WebGL core, then add the diagnostic brush layer on the same canvas.
  const composite = smallContext(w, h);
  composite.drawImage(els.gl, 0, 0, w, h);
  const coreData = composite.getImageData(0, 0, w, h).data;
  if (els.vectorToggle.checked) composite.drawImage(els.vector, 0, 0, w, h);
  const overlay = composite.getImageData(0, 0, w, h).data;
  els.mae.textContent = `${rounded(meanAbsError(coreData, reference), 2)} / ${
    rounded(meanAbsError(overlay, reference), 2)}`;
}

function render(time, forceMeter = false) {
  if (!state.meta || !state.renderer) return;
  const meta = state.meta;
  const clamped = Math.max(0, Math.min(meta.duration - 1 / meta.fps, time));
  state.time = clamped;
  const frame = frameAtTime(clamped, meta.fps, meta.frameCount);
  const cell = state.renderer.render(frame);
  renderCommands(els.vector.getContext("2d"), meta.commands,
                 clamped, meta.width, meta.height);

  const refCtx = els.reference.getContext("2d");
  refCtx.drawImage(state.sheetImage, cell.sx, cell.sy,
    cell.cellWidth, cell.cellHeight, 0, 0, meta.width, meta.height);

  if (frame !== state.lastFrame || forceMeter) {
    state.lastFrame = frame;
    els.frameLabel.textContent = `FRAME ${String(frame).padStart(3, "0")}`;
    els.frameValue.textContent = `${frame} / ${meta.frameCount - 1}`;
    els.time.textContent = `${rounded(clamped, 3)} / ${rounded(meta.duration, 3)} s`;
    els.seek.value = String(clamped);
    const index = cell.index;
    els.displacement.textContent = `${rounded(meta.motionSeries.displacement[index] || 0, 2)} px`;
    els.deformation.textContent = rounded(meta.motionSeries.deformation[index] || 0, 3);
    els.sync.textContent = `${Math.abs(frame - meta.keyframes[cell.index])} f / 0.0 ms`;
    drawChart(meta, clamped);
    for (const command of meta.commands.slice(0, 12)) {
      const m = command.motion[Math.max(0, Math.min(index, command.motion.length - 1))];
      const node = $(`motion-${command.id}`);
      if (node && m) node.textContent = `${rounded(Math.hypot(m[0], m[1]), 2)}px`;
    }
  }
  if (forceMeter) state.lastMeterAt = -1;
  measureQuality(meta, clamped, cell);
}

function loop(now) {
  if (!state.meta) return;
  const elapsed = state.lastLoopAt ? (now - state.lastLoopAt) / 1000 : 0;
  state.lastLoopAt = now;
  if (state.playing) {
    state.time += Math.min(elapsed, 0.1);
    if (state.time >= state.meta.duration - 1 / state.meta.fps) state.time = 0;
    render(state.time);
  }
  requestAnimationFrame(loop);
}

async function init() {
  try {
    const response = await fetch(assetUrl("timeline.json"));
    if (!response.ok) throw new Error(`timeline.json ${response.status}`);
    const meta = await response.json();
    state.meta = meta;
    fitCanvas(els.vector, meta.width, meta.height);
    fitCanvas(els.reference, meta.width, meta.height);
    els.seek.max = String(meta.duration);
    els.locked.textContent = `${rounded(meta.base.staticPixelRatio * 100, 1)}%`;
    renderCommandList(meta);
    drawChart(meta, 0);

    state.sheetImage = new Image();
    state.sheetImage.src = assetUrl(meta.sheet.file);
    await state.sheetImage.decode();
    state.renderer = await loadRenderer(els.gl, meta);
    els.status.textContent = `${meta.frameCount} 帧 · ${meta.commands.length} 笔触 · WebGL LOCK=ON`;
    els.status.classList.remove("error");

    window.__chronobrush = {
      meta,
      renderer: state.renderer,
      setTime: (time) => {
        state.playing = false;
        els.play.textContent = "▶";
        state.lastLoopAt = 0;
        render(time, true);
        return window.__chronobrush.diagnostic;
      },
      get diagnostic() {
        return {
          frame: state.lastFrame,
          keyframe: sheetCell(meta, state.lastFrame).index,
          baseTextureUpdates: state.renderer.baseTextureUpdates,
          frameSheetTextureUpdates: 0,
          time: state.time,
          mae: els.mae.textContent,
          sync: els.sync.textContent,
          vectorEnabled: els.vectorToggle.checked,
          splitEnabled: els.splitToggle.checked,
          runtimeVideoFiles: 0,
        };
      },
    };
    render(0, true);
    requestAnimationFrame(loop);
  } catch (error) {
    els.status.textContent = `初始化失败：${error.message}`;
    els.status.classList.add("error");
    console.error(error);
  }
}

els.play.addEventListener("click", () => {
  state.playing = !state.playing;
  els.play.textContent = state.playing ? "❚❚" : "▶";
  state.lastLoopAt = 0;
});

els.seek.addEventListener("input", () => {
  state.playing = false;
  els.play.textContent = "▶";
  state.lastLoopAt = 0;
  render(Number(els.seek.value), true);
});

function step(delta) {
  state.playing = false;
  els.play.textContent = "▶";
  state.lastLoopAt = 0;
  const meta = state.meta;
  const frame = frameAtTime(state.time, meta.fps, meta.frameCount);
  render((frame + delta) / meta.fps, true);
}

els.stepBack.addEventListener("click", () => step(-1));
els.stepForward.addEventListener("click", () => step(1));
els.vectorToggle.addEventListener("change", () => {
  els.stage.classList.toggle("no-vector", !els.vectorToggle.checked);
  render(state.time, true);
});
els.splitToggle.addEventListener("change", () => {
  els.stage.classList.toggle("split", els.splitToggle.checked);
  els.stageCaption.textContent = els.splitToggle.checked
    ? "左：重建 · 右：参考" : "重建画面";
});

init();
