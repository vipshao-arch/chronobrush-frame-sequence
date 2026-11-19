import { interpolateState, transformPoints } from "./timeline.js";

const TAU = Math.PI * 2;

function styleOf(state) {
  const [r, g, b] = state.color;
  return {
    stroke: `rgba(${r},${g},${b},${state.alpha})`,
    fill: `rgba(${r},${g},${b},${state.alpha * 0.35})`,
  };
}

export function drawEllipse(ctx, points, state, width) {
  const p = transformPoints(points, state);
  const cx = p.reduce((sum, q) => sum + q[0], 0) / p.length;
  const cy = p.reduce((sum, q) => sum + q[1], 0) / p.length;
  const rx = Math.max(3, Math.abs(p[2][0] - p[0][0]) / 2);
  const ry = Math.max(2, Math.abs(p[3][1] - p[1][1]) / 2);
  const { stroke, fill } = styleOf(state);
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, state.rotation, 0, TAU);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.strokeStyle = stroke;
  ctx.lineWidth = width;
  ctx.stroke();
}

export function drawLine(ctx, points, state, width) {
  const p = transformPoints(points, state);
  const { stroke } = styleOf(state);
  ctx.beginPath();
  ctx.moveTo(p[0][0], p[0][1]);
  for (let i = 1; i < p.length; i++) ctx.lineTo(p[i][0], p[i][1]);
  ctx.strokeStyle = stroke;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.stroke();
}

export function drawBezier(ctx, points, state, width) {
  const p = transformPoints(points, state);
  const { stroke } = styleOf(state);
  ctx.beginPath();
  ctx.moveTo(p[0][0], p[0][1]);
  ctx.bezierCurveTo(p[1][0], p[1][1], p[2][0], p[2][1],
                    p[3][0], p[3][1]);
  ctx.strokeStyle = stroke;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.stroke();
}

export function drawBrush(ctx, points, state, width) {
  const p = transformPoints(points, state);
  const { stroke } = styleOf(state);
  ctx.beginPath();
  ctx.moveTo(p[0][0], p[0][1]);
  ctx.quadraticCurveTo(p[1][0], p[1][1], p[2][0], p[2][1]);
  ctx.quadraticCurveTo(p[2][0], p[2][1], p[3][0], p[3][1]);
  ctx.strokeStyle = stroke;
  ctx.lineWidth = width * 1.7;
  ctx.lineCap = "round";
  ctx.stroke();
  for (const [x, y] of p) {
    ctx.beginPath();
    ctx.arc(x, y, width * 0.48, 0, TAU);
    ctx.fillStyle = stroke;
    ctx.fill();
  }
}

const DRAWERS = {
  ellipse: drawEllipse,
  line: drawLine,
  bezier: drawBezier,
  brush: drawBrush,
};

export function renderCommands(ctx, commands, time, width, height) {
  ctx.clearRect(0, 0, width, height);
  ctx.save();
  ctx.globalCompositeOperation = "source-over";
  for (const command of commands) {
    if (command.locked) continue;
    const state = interpolateState(command.states, time);
    if (!state) continue;
    (DRAWERS[command.type] || drawLine)(ctx, command.points, state,
                                        command.width);
  }
  ctx.restore();
}
