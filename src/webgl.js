import { sheetCell } from "./timeline.js";
import { assetUrl } from "./assets.js";

const VERT = `
attribute vec2 a_position;
varying vec2 v_uv;
void main() {
  v_uv = (a_position + 1.0) * 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}`;

const FRAG = `
precision highp float;
varying vec2 v_uv;
uniform sampler2D u_base;
uniform sampler2D u_reference;
uniform sampler2D u_mask;
uniform vec2 u_cell;
uniform vec2 u_grid;

void main() {
  // WebGL's clip-space Y is bottom-up; images are uploaded top-down.
  vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
  vec2 cell = (u_cell + uv) / u_grid;
  vec3 lockedBase = texture2D(u_base, uv).rgb;
  vec3 referenceFrame = texture2D(u_reference, cell).rgb;
  float motion = texture2D(u_mask, uv).r;
  // The low-motion base remains byte-for-byte unchanged across all frames.
  float replacement = smoothstep(0.055, 0.145, motion);
  gl_FragColor = vec4(mix(lockedBase, referenceFrame, replacement), 1.0);
}`;

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) || "Shader compile failed");
  }
  return shader;
}

async function loadImage(url) {
  const image = new Image();
  image.decoding = "async";
  image.src = url;
  await image.decode();
  return image;
}

function makeTexture(gl, image, unit) {
  const texture = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
  return texture;
}

export class LockedWebGLRenderer {
  constructor(canvas, meta, images) {
    this.canvas = canvas;
    this.meta = meta;
    this.baseTextureUpdates = 0;
    this.canvas.width = meta.width;
    this.canvas.height = meta.height;
    const gl = canvas.getContext("webgl", {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: true,
      premultipliedAlpha: false,
    });
    if (!gl) throw new Error("当前浏览器不支持 WebGL");
    const max = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    const sheetW = meta.sheet.cols * meta.sheet.cellWidth;
    const sheetH = meta.sheet.rows * meta.sheet.cellHeight;
    if (max < sheetW || max < sheetH) {
      throw new Error(`参考帧贴图 ${sheetW}×${sheetH} 超出 MAX_TEXTURE_SIZE ${max}`);
    }
    this.gl = gl;
    this.program = gl.createProgram();
    gl.attachShader(this.program, compile(gl, gl.VERTEX_SHADER, VERT));
    gl.attachShader(this.program, compile(gl, gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(this.program) || "Program link failed");
    }
    gl.useProgram(this.program);

    const quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW);
    const position = gl.getAttribLocation(this.program, "a_position");
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);

    this.textures = {
      base: makeTexture(gl, images.base, 0),
      sheet: makeTexture(gl, images.sheet, 1),
      mask: makeTexture(gl, images.mask, 2),
    };
    gl.uniform1i(gl.getUniformLocation(this.program, "u_base"), 0);
    gl.uniform1i(gl.getUniformLocation(this.program, "u_reference"), 1);
    gl.uniform1i(gl.getUniformLocation(this.program, "u_mask"), 2);
    gl.uniform2f(gl.getUniformLocation(this.program, "u_grid"),
                 meta.sheet.cols, meta.sheet.rows);
    this.cellLocation = gl.getUniformLocation(this.program, "u_cell");
    gl.viewport(0, 0, meta.width, meta.height);
  }

  render(frame) {
    const cell = sheetCell(this.meta, frame);
    this.gl.uniform2f(this.cellLocation, cell.col, cell.row);
    this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);
    return cell;
  }
}

export async function loadRenderer(canvas, meta) {
  const [base, sheet, mask] = await Promise.all([
    loadImage(assetUrl(meta.base.file)),
    loadImage(assetUrl(meta.sheet.file)),
    loadImage(assetUrl(meta.mask.file)),
  ]);
  return new LockedWebGLRenderer(canvas, meta, { base, sheet, mask });
}
