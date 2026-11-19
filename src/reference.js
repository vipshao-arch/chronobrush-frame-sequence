import { sparseFrameInfo } from "./timeline.js";
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
uniform sampler2D u_keyframes;
uniform sampler2D u_flow;
uniform sampler2D u_mask;
uniform sampler2D u_base;
uniform vec2 u_resolution;
uniform vec2 u_keyCell;
uniform vec2 u_flowCell;
uniform vec2 u_flowCellSize;
uniform vec2 u_keyAtlasSize;
uniform vec2 u_flowAtlasSize;
uniform vec2 u_flowDecode;
uniform float u_fullReference;

void main() {
  vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
  vec3 lockedBase = texture2D(u_base, uv).rgb;

  vec2 flowPixel = u_flowCell * u_flowCellSize +
    clamp(uv * u_flowCellSize, vec2(0.5), u_flowCellSize - 0.5);
  vec2 packedFlow = texture2D(u_flow, flowPixel / u_flowAtlasSize).rg;
  vec2 displacement =
    (packedFlow * 255.0 - u_flowDecode.y) * u_flowDecode.x;
  vec2 warpedPixel = clamp(uv * u_resolution + displacement,
                           vec2(0.5), u_resolution - 0.5);
  vec2 keyPixel = u_keyCell * u_resolution + warpedPixel;
  vec3 referenceFrame =
    texture2D(u_keyframes, keyPixel / u_keyAtlasSize).rgb;

  float motion = texture2D(u_mask, uv).r;
  float replacement = smoothstep(0.055, 0.145, motion);
  replacement = mix(replacement, 1.0, u_fullReference);
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

export class ReferenceRenderer {
  constructor(canvas, meta, images) {
    this.meta = meta;
    this.baseTextureUpdates = 0;
    canvas.width = meta.width;
    canvas.height = meta.height;
    const gl = canvas.getContext("webgl", {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: true,
      premultipliedAlpha: false,
    });
    if (!gl) throw new Error("当前浏览器不支持 WebGL");
    const rf = meta.referenceFrames;
    const keyAtlasWidth = rf.keyframeCols * rf.keyframeWidth;
    const keyAtlasHeight = rf.keyframeRows * rf.keyframeHeight;
    const flowAtlasWidth = rf.flowCols * rf.flowCellWidth;
    const flowAtlasHeight = rf.flowRows * rf.flowCellHeight;
    const maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    if (maxTextureSize < Math.max(keyAtlasWidth, keyAtlasHeight,
                                  flowAtlasWidth, flowAtlasHeight)) {
      throw new Error(`参考贴图集超出 MAX_TEXTURE_SIZE ${maxTextureSize}`);
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
      keyframes: makeTexture(gl, images.keyframes, 1),
      flow: makeTexture(gl, images.flow, 2),
      mask: makeTexture(gl, images.mask, 3),
    };
    gl.uniform1i(gl.getUniformLocation(this.program, "u_base"), 0);
    gl.uniform1i(gl.getUniformLocation(this.program, "u_keyframes"), 1);
    gl.uniform1i(gl.getUniformLocation(this.program, "u_flow"), 2);
    gl.uniform1i(gl.getUniformLocation(this.program, "u_mask"), 3);
    gl.uniform2f(gl.getUniformLocation(this.program, "u_resolution"),
                 meta.width, meta.height);
    gl.uniform2f(gl.getUniformLocation(this.program, "u_flowCellSize"),
                 rf.flowCellWidth, rf.flowCellHeight);
    gl.uniform2f(gl.getUniformLocation(this.program, "u_keyAtlasSize"),
                 keyAtlasWidth, keyAtlasHeight);
    gl.uniform2f(gl.getUniformLocation(this.program, "u_flowAtlasSize"),
                 flowAtlasWidth, flowAtlasHeight);
    gl.uniform2f(gl.getUniformLocation(this.program, "u_flowDecode"),
                 rf.flowScale, rf.flowOffset);
    this.keyCellLocation = gl.getUniformLocation(this.program, "u_keyCell");
    this.flowCellLocation = gl.getUniformLocation(this.program, "u_flowCell");
    this.fullReferenceLocation =
      gl.getUniformLocation(this.program, "u_fullReference");
    gl.viewport(0, 0, meta.width, meta.height);
  }

  render(frame, fullReference = false) {
    const safeFrame = Math.max(0,
      Math.min(this.meta.frameCount - 1, Math.round(frame)));
    const info = sparseFrameInfo(this.meta, safeFrame);
    this.gl.uniform2f(this.keyCellLocation, info.keyCell[0], info.keyCell[1]);
    this.gl.uniform2f(this.flowCellLocation,
                      info.flowCell[0], info.flowCell[1]);
    this.gl.uniform1f(this.fullReferenceLocation, fullReference ? 1 : 0);
    this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);
    this.gl.flush();
    return info;
  }
}

export async function loadImages(meta) {
  const rf = meta.referenceFrames;
  const files = [meta.base.file, meta.mask.file, rf.keyframeFile, rf.flowFile];
  const images = await Promise.all(files.map(async (file) => {
    const image = new Image();
    image.decoding = "async";
    image.src = assetUrl(file);
    await image.decode();
    return image;
  }));
  return {
    base: images[0], mask: images[1],
    keyframes: images[2], flow: images[3],
  };
}
