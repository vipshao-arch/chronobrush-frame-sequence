# ChronoBrush · 逐帧 WebP 数据版

这个仓库是参考帧驱动笔触重建的无 MP4 版本。原视频只在本地预分析阶段读取；发布后的页面和独立资源包只包含 WebP 帧数据、运动掩膜、静态基底和 JSON 时间轴。运行时由时间轴计算帧号，从 WebP 帧集中选择对应画面，再用同一时刻的状态绘制椭圆、线段、贝塞尔和笔触。

## 在线查看

<https://vipshao-arch.github.io/chronobrush-frame-sequence/>

页面提供播放、逐帧步进、拖动时间轴、指令层开关、左右分屏对比，以及核心画面和叠加笔触的 MAE 读数。

## 本地运行

```bash
npm install
npm run dev
```

`npm test` 验证时间轴映射、状态插值、贴图格坐标和预分析数据；`npm run build` 生成 `docs/` 静态站点。

## 资源数据

`public/assets/` 和独立压缩包 `chronobrush-runtime-assets.zip` 都只包含这 4 个文件：

| 文件 | 用途 |
| --- | --- |
| `base.webp` | 锁定的 WebGL 静态基底，初始化后不更新 |
| `reference.webp` | 145 帧 WebP 贴图集，每个格子是一帧参考画面 |
| `motion-mask.webp` | 运动区域掩膜，决定静态基底与当前帧的混合 |
| `timeline.json` | 96 条绘图指令、逐帧颜色、位移、旋转和形变状态 |

`timeline.json` 中每条指令都有 4 个控制点、绘制宽度以及 145 个状态。`src/timeline.js` 按时间查找状态并插值；`src/draw.js` 调用绘图函数；`src/webgl.js` 在着色器中混合基底和当前帧格子。帧集是常驻 WebGL 纹理，播放过程中只切换格子坐标，不上传新纹理，也不读取任何视频文件。

重新生成资源和资源包：

```bash
python -m pip install opencv-python-headless numpy pillow
python tools/build_assets.py --video "参考.mp4" --out public/assets
python package_assets.py
```

第一条命令需要本地原视频，产出的资源目录和 ZIP 均不含 MP4。校验清单为 `chronobrush-runtime-assets.sha256.json`。

## 还原边界

核心画面在运动区域由预分析帧驱动，因此能逼近参考帧；静态区域来自帧 0 的无损基底，不随帧漂移。帧集使用 735×315 的格子，显示时放大到 1470×630；半透明诊断笔触用于展示每条指令的运动参数，会略微提高叠加层的像素误差，可关闭。此版本保留逐帧像素还原和时间参数可视化，但不包含任何原始视频容器或视频解码器。
