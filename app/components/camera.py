"""Camera with a head-and-shoulders guide and live face checks (in the browser).

The outline is red, with a hint, until the face passes the same checks as an uploaded photo
(skiniq.vision.quality): one face, framed head and shoulders inside the outline, head
straight and facing the camera, nothing covering it (no sunglasses, glasses, mask, hand or
hair over an eye or one side), and evenly lit. Only after three frames in a row pass does it
turn green and enable "Take photo". The thresholds come from quality.settings(), so the guide
and the server check agree; the captured photo is exactly the picture inside the frame and is
checked again on the server.

Face detection runs in the browser on the CPU with YuNet (the same detector as the server
check) and ONNX Runtime Web, bundled in app/static/facedetect and served by Streamlit static
serving. It needs no WebGL. If the detector cannot load, the guide stays grey, the photo can
still be taken (the status line says why), and the server check decides.

Result, kept in st.session_state[key]["photo"]: a JPEG data URL, or None after "Retake".
"""

from __future__ import annotations

import base64

import streamlit as st

from skiniq.vision.quality import settings

STATIC_BASE = "app/static/facedetect/"

_HTML = """
<div class="cam">
  <div class="stage">
    <video playsinline muted></video>
    <img class="still" alt="Captured photo" hidden />
    <svg class="guide" viewBox="0 0 640 480" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
      <defs>
        <mask id="cutout">
          <rect width="640" height="480" fill="white" />
          <ellipse cx="320" cy="205" rx="118" ry="152" fill="black" />
        </mask>
      </defs>
      <rect class="shade" width="640" height="480" mask="url(#cutout)" />
      <ellipse class="outline" cx="320" cy="205" rx="118" ry="152" />
      <path class="outline shoulders" d="M 150 480 C 160 405, 230 372, 268 360 M 372 360 C 410 372, 480 405, 490 480" />
    </svg>
    <div class="status">Starting camera…</div>
  </div>
  <div class="controls">
    <button type="button" class="shoot" disabled>Take photo</button>
    <button type="button" class="retake" hidden>Retake</button>
  </div>
</div>
"""

_CSS = """
.cam { font-family: var(--st-font); color: var(--st-text-color); }
.stage { position: relative; width: 100%; max-width: 560px; aspect-ratio: 4 / 3; margin: 0 auto;
         border-radius: var(--st-base-radius); overflow: hidden; background: #111; }
video, .still { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }
video { transform: scaleX(-1); }
.guide { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
.shade { fill: rgba(0, 0, 0, 0.45); }
.outline { fill: none; stroke: #9aa0a6; stroke-width: 5; stroke-dasharray: 14 9; transition: stroke 0.2s; }
.cam.red .outline { stroke: #e5484d; stroke-dasharray: none; }
.cam.green .outline { stroke: #30a46c; stroke-dasharray: none; }
.cam.captured .guide { display: none; }
.status { position: absolute; left: 0; right: 0; bottom: 0; padding: 0.5rem 0.75rem; text-align: center;
          font-size: 0.95rem; color: white; background: rgba(0, 0, 0, 0.55); }
.cam.red .status { background: rgba(160, 30, 35, 0.8); }
.cam.green .status { background: rgba(25, 110, 70, 0.85); }
.controls { display: flex; gap: 0.5rem; justify-content: center; margin-top: 0.75rem; }
button { font: inherit; padding: 0.45rem 1.1rem; border-radius: var(--st-base-radius); cursor: pointer;
         border: 1px solid var(--st-border-color); background: var(--st-secondary-background-color);
         color: var(--st-text-color); }
button.shoot:not(:disabled) { background: var(--st-primary-color); border-color: var(--st-primary-color); color: white; }
button:disabled { opacity: 0.5; cursor: not-allowed; }
"""

_JS = """
const instances = new WeakMap();

// The visible picture is analysed as a 640x480 frame (the stage is 4:3, video uses object-fit: cover).
const FRAME_W = 640, FRAME_H = 480;
const PASS_FRAMES = 3;           // frames in a row that must pass before the outline turns green

const LOAD_TIMEOUT_MS = 25000;   // ~11 MB of WebAssembly on first load
const MAX_FRAME_ERRORS = 8;      // consecutive failed frames before falling back
const IN_W = 320, IN_H = 256;    // detector input (multiples of 32); frames are letterboxed into it
const SCORE_MIN = 0.7, NMS_IOU = 0.3;   // same face score threshold as the server check

let detectorPromise = null;
let detectorError = "";
function loadDetector(base) {
  if (!detectorPromise) {
    const load = (async () => {
      const root = new URL(base, document.baseURI).href;
      const ort = await import(root + "ort.wasm.bundle.min.mjs");
      ort.env.wasm.wasmPaths = root;
      ort.env.wasm.numThreads = 1;  // no SharedArrayBuffer needed
      const session = await ort.InferenceSession.create(root + "yunet_2023mar_dynamic.onnx",
        { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
      return makeDetector(ort, session);
    })();
    const timeout = new Promise((_, reject) =>
      setTimeout(() => reject(new Error("timed out loading the face detector")), LOAD_TIMEOUT_MS));
    detectorPromise = Promise.race([load, timeout]).catch((e) => {
      detectorError = (e && e.message) ||
        (typeof Event !== "undefined" && e instanceof Event ? "a detector file could not be loaded" : String(e));
      console.warn("SkinIQ face detector unavailable:", e);
      detectorPromise = null;  // allow a retry next time the camera starts
      return null;
    });
  }
  return detectorPromise;
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[0] + a[2], b[0] + b[2]), y2 = Math.min(a[1] + a[3], b[1] + b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a[2] * a[3] + b[2] * b[3] - inter);
}

// YuNet (OpenCV Zoo, 2023mar) on the 640x480 frame canvas. Returns faces as YuNet rows, like OpenCV:
// [x, y, w, h, eye1 x, y, eye2 x, y, nose x, y, mouth1 x, y, mouth2 x, y, score] in frame pixels.
function makeDetector(ort, session) {
  const canvas = document.createElement("canvas");
  canvas.width = IN_W; canvas.height = IN_H;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const input = new Float32Array(3 * IN_W * IN_H);
  return async (frame) => {
    const sc = Math.min(IN_W / frame.width, IN_H / frame.height);
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, IN_W, IN_H);
    ctx.drawImage(frame, 0, 0, frame.width * sc, frame.height * sc);
    const px = ctx.getImageData(0, 0, IN_W, IN_H).data, n = IN_W * IN_H;
    for (let i = 0; i < n; i++) {  // planar BGR, 0-255 (as cv2.dnn.blobFromImage)
      input[i] = px[i * 4 + 2]; input[n + i] = px[i * 4 + 1]; input[2 * n + i] = px[i * 4];
    }
    const out = await session.run({ input: new ort.Tensor("float32", input, [1, 3, IN_H, IN_W]) });
    const cands = [];
    for (const st of [8, 16, 32]) {
      const cols = IN_W / st, cls = out["cls_" + st].data, obj = out["obj_" + st].data;
      const bb = out["bbox_" + st].data, kps = out["kps_" + st].data;
      for (let i = 0; i < cls.length; i++) {
        const score = Math.sqrt(Math.min(Math.max(cls[i], 0), 1) * Math.min(Math.max(obj[i], 0), 1));
        if (score < SCORE_MIN) continue;
        const r = Math.floor(i / cols), c = i % cols;
        const cx = (c + bb[i * 4]) * st, cy = (r + bb[i * 4 + 1]) * st;
        const w = Math.exp(bb[i * 4 + 2]) * st, h = Math.exp(bb[i * 4 + 3]) * st;
        const row = [(cx - w / 2) / sc, (cy - h / 2) / sc, w / sc, h / sc];
        for (let k = 0; k < 5; k++) row.push((c + kps[i * 10 + 2 * k]) * st / sc, (r + kps[i * 10 + 2 * k + 1]) * st / sc);
        row.push(score);
        cands.push(row);
      }
    }
    cands.sort((a, b) => b[14] - a[14]);
    const kept = [];
    for (const f of cands) if (kept.every((k) => iou(k, f) < NMS_IOU)) kept.push(f);
    // Like the server check: a much smaller face in the background is tolerated
    kept.sort((a, b) => b[2] * b[3] - a[2] * a[3]);
    const big = kept.length ? kept[0][2] * kept[0][3] : 0;
    return kept.filter((f) => f[2] * f[3] > 0.25 * big || f === kept[0]);
  };
}

// ---- the same checks as skiniq/vision/quality.py (thresholds passed in from Python) ----------

function landmarks(f) {
  let e1 = [f[4], f[5]], e2 = [f[6], f[7]];
  const [left, right] = e1[0] <= e2[0] ? [e1, e2] : [e2, e1];
  return { left, right, nose: [f[8], f[9]], mouth: [(f[10] + f[12]) / 2, (f[11] + f[13]) / 2] };
}

function framingProblems(f, width, height, F) {
  const [x, y, w, h] = f, codes = [], m = F.edge_margin;
  if (x < m * width || y < m * height || x + w > (1 - m) * width || y + h > (1 - m) * height) codes.push("cut_off");
  const faceH = h / height;
  if (faceH < F.min_face_height) codes.push("too_far");
  else if (faceH > F.max_face_height || (y + h) / height > F.max_chin) {
    codes.push("too_close");
    if (codes[0] === "cut_off") codes.shift();   // an oversized face runs off the edges: moving back fixes both
  }
  const cx = (x + w / 2) / width, cy = (y + h / 2) / height;
  if (!codes.includes("cut_off") && !(cx >= F.min_centre_x && cx <= F.max_centre_x &&
                                      cy >= F.min_centre_y && cy <= F.max_centre_y)) codes.push("off_centre");
  const { left, right, nose } = landmarks(f);
  const d = Math.hypot(right[0] - left[0], right[1] - left[1]);
  if (d > 0) {
    if (Math.abs(Math.atan2(right[1] - left[1], right[0] - left[0]) * 180 / Math.PI) > F.max_roll_deg) codes.push("tilted");
    if (Math.abs(nose[0] - (left[0] + right[0]) / 2) / d > F.max_yaw) codes.push("turned");
  }
  return codes;
}

const CW = 192, CH = 224;
const CANON = [[64, 80], [128, 80], [96, 152]];   // eyes and mouth centre in the aligned face
const PATCHES = {
  eye_l: [64, 80, 34, 18], eye_r: [128, 80, 34, 18], brow_mid: [96, 62, 22, 14], forehead: [96, 34, 64, 18],
  cheek_l: [58, 122, 26, 24], cheek_r: [134, 122, 26, 24], nose: [96, 118, 18, 16], mouth: [96, 152, 40, 14],
  chin: [96, 188, 36, 16], bridge: [96, 82, 14, 14], under_l: [64, 101, 34, 10], under_r: [128, 101, 34, 10],
};

// Affine transform taking three source points to three destination points: [a, b, c, d, e, f] for setTransform.
function affine(src, dst) {
  const [[x0, y0], [x1, y1], [x2, y2]] = src;
  const det = x0 * (y1 - y2) - y0 * (x1 - x2) + (x1 * y2 - x2 * y1);
  const solve = (v0, v1, v2) => [
    (v0 * (y1 - y2) - y0 * (v1 - v2) + (v1 * y2 - v2 * y1)) / det,
    (x0 * (v1 - v2) - v0 * (x1 - x2) + (x1 * v2 - x2 * v1)) / det,
    (x0 * (y1 * v2 - y2 * v1) - y0 * (x1 * v2 - x2 * v1) + v0 * (x1 * y2 - x2 * y1)) / det,
  ];
  const [a, c, e] = solve(dst[0][0], dst[1][0], dst[2][0]);
  const [b, d, f] = solve(dst[0][1], dst[1][1], dst[2][1]);
  return [a, b, c, d, e, f];
}

function srgbToLab(r, g, b) {   // D65, as OpenCV's RGB2Lab: L* 0-100, a*, b*
  const lin = (v) => { v /= 255; return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
  const R = lin(r), G = lin(g), B = lin(b);
  const X = (0.412453 * R + 0.357580 * G + 0.180423 * B) / 0.950456;
  const Y = 0.212671 * R + 0.715160 * G + 0.072169 * B;
  const Z = (0.019334 * R + 0.119193 * G + 0.950227 * B) / 1.088754;
  const fn = (t) => t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116;
  const fx = fn(X), fy = fn(Y), fz = fn(Z);
  return [Y > 0.008856 ? 116 * fy - 16 : 903.3 * Y, 500 * (fx - fy), 200 * (fy - fz)];
}

function median(values) { const v = Float32Array.from(values).sort(); const n = v.length; return n % 2 ? v[(n - 1) / 2] : (v[n / 2 - 1] + v[n / 2]) / 2; }
function mean(values) { let s = 0; for (const x of values) s += x; return s / values.length; }
function std(values) { const m = mean(values); let s = 0; for (const x of values) s += (x - m) * (x - m); return Math.sqrt(s / values.length); }

function faceVisibility(frame, f, canon) {
  const { left, right, mouth } = landmarks(f);
  const ctx = canon.getContext("2d", { willReadFrequently: true });
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#808080"; ctx.fillRect(0, 0, CW, CH);
  ctx.setTransform(...affine([left, right, mouth], CANON));
  ctx.drawImage(frame, 0, 0);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  const px = ctx.getImageData(0, 0, CW, CH).data;
  const n = CW * CH, L = new Float32Array(n), A = new Float32Array(n), B = new Float32Array(n), G = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const r = px[4 * i], g = px[4 * i + 1], b = px[4 * i + 2];
    [L[i], A[i], B[i]] = srgbToLab(r, g, b);
    G[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }
  const region = (name, arr) => {
    const [x, y, w, h] = PATCHES[name], out = [];
    for (let yy = y - (h >> 1); yy < y + (h >> 1); yy++) for (let xx = x - (w >> 1); xx < x + (w >> 1); xx++) out.push(arr[yy * CW + xx]);
    return out;
  };
  const med = {};
  for (const name in PATCHES) med[name] = [median(region(name, L)), median(region(name, A)), median(region(name, B))];
  const med3 = (names) => [0, 1, 2].map((k) => median(names.map((nm) => med[nm][k])));
  const skin = med3(["cheek_l", "cheek_r", "nose"]), upper = med3(["brow_mid", "forehead"]), lower = med3(["nose", "mouth", "chin"]);
  const skinL = Math.max(skin[0], 1), m = {};
  for (const s of ["l", "r"]) {
    const eye = region("eye_" + s, L);
    m["eye_" + s + "_rel"] = mean(eye) / skinL;
    m["eye_" + s + "_dark"] = mean(eye.map((v) => (v < 0.6 * skinL ? 1 : 0)));
    m["eye_" + s + "_contrast"] = std(eye) / skinL;
  }
  const dist = (p, q) => Math.hypot(p[1] - q[1], p[2] - q[2]);
  m.lower_colour_diff = dist(lower, upper);
  m.lip_red = med.mouth[1] - upper[1];
  m.mouth_texture = std(region("mouth", L)) / Math.max(upper[0], 1);
  m.cheek_colour_diff = dist(med.cheek_l, med.cheek_r);
  m.cheek_light_diff = Math.abs(med.cheek_l[0] - med.cheek_r[0]);
  // Sobel 3x3 on the grey face (as cv2.Sobel, borders replicated)
  const gx = new Float32Array(n), gy = new Float32Array(n), grad = new Float32Array(n);
  const at = (x, y) => G[Math.min(CH - 1, Math.max(0, y)) * CW + Math.min(CW - 1, Math.max(0, x))];
  for (let y = 0; y < CH; y++) for (let x = 0; x < CW; x++) {
    const i = y * CW + x;
    gx[i] = Math.abs(at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2 * at(x - 1, y) - at(x - 1, y + 1));
    gy[i] = Math.abs(at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2 * at(x, y - 1) - at(x + 1, y - 1));
    grad[i] = Math.hypot(gx[i], gy[i]);
  }
  const cheekDetail = (mean(region("cheek_l", grad)) + mean(region("cheek_r", grad))) / 2 + 4;
  m.bridge_edges = mean(region("bridge", grad)) / cheekDetail;
  m.under_edges = Math.max(mean(region("under_l", gy)), mean(region("under_r", gy))) / cheekDetail;
  const half = (x0, x1) => { let s = 0, c = 0; for (let y = 50; y < 200; y++) for (let x = x0; x < x1; x++) { s += grad[y * CW + x]; c++; } return s / c; };
  const hl = half(30, 88), hr = half(104, 162);
  m.half_texture_ratio = Math.min(hl, hr) / Math.max(hl, hr, 1e-6);
  let clipped = 0, crushed = 0, total = 0;
  for (const name of ["cheek_l", "cheek_r", "nose", "forehead"]) {
    const [x, y, w, h] = PATCHES[name];
    for (let yy = y - (h >> 1); yy < y + (h >> 1); yy++) for (let xx = x - (w >> 1); xx < x + (w >> 1); xx++) {
      const i = 4 * (yy * CW + xx), r = px[i], g = px[i + 1], b = px[i + 2];
      if (Math.min(r, g, b) >= 250) clipped++;
      if (Math.max(r, g, b) <= 25) crushed++;
      total++;
    }
  }
  m.clipped = clipped / total; m.crushed = crushed / total;
  return m;
}

function visibilityProblems(m, V, LI) {
  const codes = [];
  if (Math.max(m.eye_l_rel, m.eye_r_rel) < V.sunglasses_max_rel && Math.min(m.eye_l_dark, m.eye_r_dark) > V.sunglasses_min_dark) codes.push("sunglasses");
  else if (m.bridge_edges > V.glasses_min_bridge_edges && m.under_edges > V.glasses_min_under_edges) codes.push("glasses");
  if (["l", "r"].some((s) => m["eye_" + s + "_contrast"] < V.eye_min_contrast && m["eye_" + s + "_rel"] > V.eye_covered_min_rel)) codes.push("eye_covered");
  if ((m.lower_colour_diff > V.mask_min_colour_diff && m.lip_red < V.mask_max_lip_red) ||
      (m.mouth_texture < V.mask_max_mouth_texture && m.lip_red < V.mask_max_lip_red_flat)) {
    codes.push("mask");
    const g = codes.indexOf("glasses"); if (g >= 0) codes.splice(g, 1);
  }
  if ((m.cheek_colour_diff > V.half_max_colour_diff || m.half_texture_ratio < V.half_min_texture_ratio) && !codes.includes("eye_covered")) codes.push("half_covered");
  if (m.crushed > LI.max_crushed) codes.push("too_dark");
  else if (m.clipped > LI.max_clipped) codes.push("too_bright");
  if (m.cheek_light_diff > LI.max_side_diff) codes.push("uneven_light");
  return codes;
}

// ---- trained models in the browser: face parts (BiSeNet face parsing) and hands (MediaPipe palm) ----
// Same crops, measurements and thresholds as skiniq/vision/face_models.py.

const PARSE = 512, CROP_SCALE = 1.7;
const L_SKIN = 1, L_LEYE = 4, L_REYE = 5, L_GLASSES = 6, L_NOSE = 10, L_MOUTH = 11, L_ULIP = 12, L_LLIP = 13, L_HAIR = 17, L_HAT = 18;
const PARTS_EVERY_MS = 600, PARTS_FRESH_MS = 2000;

let partsPromise = null;
function loadPartsModels(base, files) {
  if (!partsPromise) {
    partsPromise = (async () => {
      const root = new URL(base, document.baseURI).href;
      const ort = await import(root + "ort.wasm.bundle.min.mjs");   // already loaded for YuNet
      const modelsRoot = new URL("app/static/photo_check/", document.baseURI).href;
      const [parse, palm, anchors] = await Promise.all([
        ort.InferenceSession.create(modelsRoot + files.face_parsing, { executionProviders: ["wasm"] }),
        ort.InferenceSession.create(modelsRoot + files.palm, { executionProviders: ["wasm"] }),
        fetch(root + "palm_anchors.json").then((r) => r.json()),
      ]);
      return { ort, parse, palm, anchors };
    })().catch((e) => { console.warn("SkinIQ face-part models unavailable:", e); return null; });
  }
  return partsPromise;
}

function cropCanvas(src, cx, cy, side, size) {
  const c = document.createElement("canvas");
  c.width = size; c.height = size;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, size, size);
  const s = size / side;
  ctx.setTransform(s, 0, 0, s, -(cx - side / 2) * s, -(cy - side / 2) * s);
  ctx.drawImage(src, 0, 0);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  return { canvas: c, ctx, s, x0: cx - side / 2, y0: cy - side / 2 };
}

async function parseFace(M, frame, f) {
  const [x, y, w, h] = f, side = Math.max(w, h) * CROP_SCALE, cx = x + w / 2, cy = y + h / 2;
  const { ctx, s, x0, y0 } = cropCanvas(frame, cx, cy, side, PARSE);
  const px = ctx.getImageData(0, 0, PARSE, PARSE).data, n = PARSE * PARSE;
  const input = new Float32Array(3 * n), mean = [0.485, 0.456, 0.406], sd = [0.229, 0.224, 0.225];
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) input[k * n + i] = (px[4 * i + k] / 255 - mean[k]) / sd[k];
  const out = await M.parse.run({ input: new M.ort.Tensor("float32", input, [1, 3, PARSE, PARSE]) }, ["output"]);
  const logits = out.output.data, labels = new Uint8Array(n), valid = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    let best = 0, bv = logits[i];
    for (let k = 1; k < 19; k++) { const v = logits[k * n + i]; if (v > bv) { bv = v; best = k; } }
    labels[i] = best;
    const fx = (i % PARSE) / s + x0, fy = Math.floor(i / PARSE) / s + y0;
    valid[i] = fx >= 0 && fy >= 0 && fx < frame.width && fy < frame.height && Math.max(px[4 * i], px[4 * i + 1], px[4 * i + 2]) > 6 ? 1 : 0;
  }
  const pts = [];
  for (let k = 0; k < 5; k++) pts.push([(f[4 + 2 * k] - x0) * s, (f[5 + 2 * k] - y0) * s]);
  return { labels, valid, pts, px };
}

function partsMeasures(P) {
  const { labels, valid, pts, px } = P;
  const [left, right] = pts[0][0] <= pts[1][0] ? [pts[0], pts[1]] : [pts[1], pts[0]];
  const iod = Math.max(Math.hypot(right[0] - left[0], right[1] - left[1]), 1);
  const lum = (i) => srgbToLab(px[4 * i], px[4 * i + 1], px[4 * i + 2]);
  const box = (cx, cy, hw, hh, fn) => {
    for (let y = Math.max(0, Math.floor(cy - hh)); y < Math.min(PARSE, Math.floor(cy + hh)); y++)
      for (let x = Math.max(0, Math.floor(cx - hw)); x < Math.min(PARSE, Math.floor(cx + hw)); x++) fn(y * PARSE + x);
  };
  const share = (cx, cy, hw, hh, ids) => {
    let hit = 0, inside = 0;
    box(cx, cy, hw, hh, (i) => { if (valid[i]) { inside++; if (ids.includes(labels[i])) hit++; } });
    return inside ? hit / inside : 0;
  };
  let skinL = [], skinA = [];
  for (let i = 0; i < labels.length; i += 3) if (labels[i] === L_SKIN && valid[i]) { const l = lum(i); skinL.push(l[0]); skinA.push(l[1]); }
  const sL = skinL.length > 50 ? median(skinL) : 50;
  const m = {}, lens = [];
  for (const [name, e] of [["left", left], ["right", right]]) {
    let eye = 0;
    box(e[0], e[1], 0.45 * iod, 0.3 * iod, (i) => {
      if (labels[i] === L_LEYE || labels[i] === L_REYE) eye++;
      if (labels[i] === L_GLASSES && valid[i]) lens.push(lum(i)[0]);
    });
    m["eye_" + name] = eye / (iod * iod);
    m["glasses_" + name] = share(e[0], e[1], 0.45 * iod, 0.3 * iod, [L_GLASSES]);
  }
  m.lens_lightness = lens.length > 30 ? median(lens) / Math.max(sL, 1) : 1;
  m.nose = share(pts[2][0], pts[2][1] - 0.15 * iod, 0.35 * iod, 0.4 * iod, [L_NOSE]);
  const mouth = [(pts[3][0] + pts[4][0]) / 2, (pts[3][1] + pts[4][1]) / 2];
  m.lips = share(mouth[0], mouth[1], 0.55 * iod, 0.35 * iod, [L_ULIP, L_LLIP, L_MOUTH]);
  m.forehead_covered = share((left[0] + right[0]) / 2, (left[1] + right[1]) / 2 - 0.75 * iod, 0.6 * iod, 0.25 * iod, [L_HAIR, L_HAT]);
  m.skin_redness = skinA.length > 50 ? median(skinA) : 0;
  return m;
}

function partsProblems(m, T) {
  const codes = [];
  if (Math.min(m.glasses_left, m.glasses_right) >= T.min_glasses_share) codes.push(m.lens_lightness < T.max_lens_lightness ? "sunglasses" : "glasses");
  else if (Math.min(m.eye_left, m.eye_right) < T.min_eye_area) codes.push("eye_hidden");
  if (m.nose < T.min_nose_share || m.lips < T.min_lips_share) codes.push("mask");
  if (m.forehead_covered > T.max_forehead_covered) codes.push("forehead_covered");
  if (m.skin_redness < T.min_skin_redness && !codes.includes("mask")) codes.push("unnatural_skin");
  return codes;
}

// MediaPipe palm detector (as skiniq/vision/vendor/mp_palmdet.py): palms in source pixels with scores.
async function palms(M, src, threshold) {
  const S = 192, h = src.height, w = src.width, ratio = Math.min(S / h, S / w);
  const rw = Math.floor(w * ratio), rh = Math.floor(h * ratio), left = Math.floor((S - rw) / 2), top = Math.floor((S - rh) / 2);
  const c = document.createElement("canvas"); c.width = S; c.height = S;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, S, S); ctx.drawImage(src, left, top, rw, rh);
  const px = ctx.getImageData(0, 0, S, S).data, input = new Float32Array(S * S * 3);
  for (let i = 0; i < S * S; i++) { input[3 * i] = px[4 * i] / 255; input[3 * i + 1] = px[4 * i + 1] / 255; input[3 * i + 2] = px[4 * i + 2] / 255; }
  const out = await M.palm.run({ input_1: new M.ort.Tensor("float32", input, [1, S, S, 3]) });
  const reg = out.Identity.data, raw = out.Identity_1.data, scale = Math.max(w, h);
  const padX = Math.floor(left / ratio), padY = Math.floor(top / ratio), cands = [];
  for (let i = 0; i < M.anchors.length; i++) {
    const score = 1 / (1 + Math.exp(-raw[i]));
    if (score < threshold) continue;
    const [ax, ay] = M.anchors[i], cx = reg[18 * i] / S + ax, cy = reg[18 * i + 1] / S + ay;
    const bw = reg[18 * i + 2] / S, bh = reg[18 * i + 3] / S;
    cands.push([(cx - bw / 2) * scale - padX, (cy - bh / 2) * scale - padY, bw * scale, bh * scale, score]);
  }
  cands.sort((a, b) => b[4] - a[4]);
  const kept = [];
  for (const b of cands) if (kept.every((k) => iou(k, b) < 0.3)) kept.push(b);
  return kept;
}

async function handOnFace(M, frame, f, H) {
  const [x, y, w, h] = f, fx0 = x - 0.1 * w, fy0 = y - 0.1 * h, fx1 = x + 1.1 * w, fy1 = y + 1.1 * h;
  const side = 2.2 * Math.max(w, h), cx = x + w / 2, cy = y + h / 2;
  const close = cropCanvas(frame, cx, cy, side, 384);
  const sources = [[frame, 1, 0, 0], [close.canvas, 1 / close.s, close.x0, close.y0]];
  for (const [src, k, ox, oy] of sources) {
    for (const [bx, by, bw, bh, score] of await palms(M, src, H.palm_score)) {
      const px0 = bx * k + ox, py0 = by * k + oy, px1 = px0 + bw * k, py1 = py0 + bh * k;
      const overlap = Math.max(0, Math.min(px1, fx1) - Math.max(px0, fx0)) * Math.max(0, Math.min(py1, fy1) - Math.max(py0, fy0));
      if (overlap >= H.min_overlap * (px1 - px0) * (py1 - py0)) return true;
    }
  }
  return false;
}

function highlights(frame) {   // L* of the brightest 5% of the frame (as face_models.highlights)
  const c = document.createElement("canvas"); c.width = 160; c.height = 120;
  const ctx = c.getContext("2d", { willReadFrequently: true }); ctx.drawImage(frame, 0, 0, 160, 120);
  const px = ctx.getImageData(0, 0, 160, 120).data, L = [];
  for (let i = 0; i < px.length; i += 4) L.push(srgbToLab(px[i], px[i + 1], px[i + 2])[0]);
  L.sort((a, b) => a - b);
  return L[Math.floor(0.95 * (L.length - 1))];
}

async function modelProblems(M, frame, f, S) {
  const P = await parseFace(M, frame, f);
  const codes = partsProblems(partsMeasures(P), S.parts);
  if (await handOnFace(M, frame, f, S.hands)) codes.push("hand");
  if (highlights(frame) < S.parts.min_highlights) codes.push("too_dark");
  return codes;
}

// Short hints for the live guide; other problems use the server check's messages.
const LIVE_HINTS = {
  too_far: "Move closer — fit your face in the outline",
  too_close: "Move back — your head and shoulders should fit the outline",
  off_centre: "Centre your face in the outline",
  cut_off: "Move back or centre your face — your whole face must be inside the outline",
  tilted: "Keep your head straight",
  turned: "Face the camera straight on",
  hand: "Move your hands away from your face",
  eye_hidden: "Keep both eyes open and uncovered",
  forehead_covered: "Your forehead is covered — pull your hair back or remove your hat",
};

function judge(faces, frame, canon, S, parts) {
  if (faces.length === 0) return ["red", "No face detected — look straight at the camera with nothing covering your face", []];
  if (faces.length > 1) return ["red", "More than one face — only you should be in the frame", ["several_faces"]];
  const f = faces[0];
  const framing = framingProblems(f, frame.width, frame.height, S.framing);
  // Measure the face once it is placed in the outline (size and position OK). Something covering the
  // face can also shift the landmarks and look like a turned or tilted head, so it is reported first.
  const placed = framing.every((c) => c === "tilted" || c === "turned");
  let codes = framing;
  if (placed) {
    const m = faceVisibility(frame, f, canon);
    let vis;
    if (parts.loading) return ["red", "Loading face checks…", ["checking"]];   // never green before the models run
    if (parts.models) {
      // trained-model checks (latest result, refreshed every ~0.6 s) plus the lighting measures
      if (!parts.codes || performance.now() - parts.time > PARTS_FRESH_MS) return ["red", "Checking your face…", ["checking"]];
      vis = [...parts.codes];
      if (m.crushed > S.lighting.max_crushed && !vis.includes("too_dark")) vis.push("too_dark");
      else if (m.clipped > S.lighting.max_clipped && !vis.includes("too_dark")) vis.push("too_bright");
      if (m.cheek_light_diff > S.lighting.max_side_diff && !vis.includes("hand")) vis.push("uneven_light");
    } else {
      vis = visibilityProblems(m, S.visibility, S.lighting);   // rule-based fallback
    }
    codes = [...vis, ...framing];
  }
  if (codes.length) return ["red", LIVE_HINTS[codes[0]] || S.messages[codes[0]], codes];
  return ["green", "Face detected — hold still and take the photo", []];
}

// The part of the video shown in the 4:3 stage (object-fit: cover), in video pixels.
function visibleRegion(video) {
  const vw = video.videoWidth, vh = video.videoHeight;
  const scale = Math.max(FRAME_W / vw, FRAME_H / vh);
  const w = FRAME_W / scale, h = FRAME_H / scale;
  return [(vw - w) / 2, (vh - h) / 2, w, h];
}

export default function (component) {
  const { data, parentElement, setStateValue } = component;
  const root = parentElement.querySelector(".cam");
  const video = parentElement.querySelector("video");
  const still = parentElement.querySelector(".still");
  const status = parentElement.querySelector(".status");
  const shoot = parentElement.querySelector(".shoot");
  const retake = parentElement.querySelector(".retake");
  let inst = instances.get(parentElement);
  if (!inst) { inst = { stream: null, timer: null, aligned: false, detector: null }; instances.set(parentElement, inst); }

  const setState = (cls, text) => {
    root.classList.remove("red", "green");
    if (cls) root.classList.add(cls);
    status.textContent = text;
  };

  const stopLoop = () => { if (inst.timer) { clearInterval(inst.timer); inst.timer = null; } };
  const stopCamera = () => {
    stopLoop();
    if (inst.stream) { inst.stream.getTracks().forEach((t) => t.stop()); inst.stream = null; }
  };

  const fallback = (why) => {
    stopLoop();
    inst.detector = null;
    console.warn("SkinIQ face guide off:", why);
    setState(null, "Face guide unavailable on this browser (" + why + "). Fit your face in the outline " +
      "and take the photo; it is checked after you take it.");
    shoot.disabled = false;
  };

  const startLoop = () => {
    stopLoop();
    const frame = document.createElement("canvas");
    frame.width = FRAME_W; frame.height = FRAME_H;
    const fctx = frame.getContext("2d", { willReadFrequently: true });
    const canon = document.createElement("canvas");
    canon.width = CW; canon.height = CH;
    let errors = 0, passes = 0;
    const startedAt = performance.now();
    setState("red", "Looking for your face…");
    let busy = false;
    // trained face-part and hand models: loaded in the background, run every PARTS_EVERY_MS
    const parts = { models: null, loading: !!data.parts_models, codes: null, time: 0, busy: false, last: 0 };
    if (data.parts_models) loadPartsModels(data.static_base, data.parts_models).then((M) => { parts.models = M; parts.loading = false; });
    const runParts = (faces) => {
      if (!parts.models || parts.busy || faces.length !== 1 || performance.now() - parts.last < PARTS_EVERY_MS) return;
      const snap = document.createElement("canvas");
      snap.width = FRAME_W; snap.height = FRAME_H;
      snap.getContext("2d").drawImage(frame, 0, 0);
      parts.busy = true; parts.last = performance.now();
      const t0 = performance.now();
      modelProblems(parts.models, snap, faces[0], data.settings)
        .then((codes) => {
          if (!parts.codes) console.info(`SkinIQ face-part check: ${Math.round(performance.now() - t0)} ms`);
          parts.codes = codes; parts.time = performance.now();
        })
        .catch((e) => { console.warn("SkinIQ face-part check failed:", e); })
        .finally(() => { parts.busy = false; });
    };
    inst.timer = setInterval(async () => {
      if (!inst.detector || busy) return;
      if (video.readyState < 2 || !video.videoWidth) {
        if (performance.now() - startedAt > 8000) setState("red", "Waiting for the camera picture…");
        return;
      }
      const detect = inst.detector;
      busy = true;
      let verdict;
      try {
        fctx.drawImage(video, ...visibleRegion(video), 0, 0, FRAME_W, FRAME_H);  // unmirrored, as captured
        const faces = await detect(frame);
        verdict = judge(faces, frame, canon, data.settings, parts);
        if (!verdict[2].length || verdict[2].every((c) => c === "checking" || !["too_far", "too_close", "off_centre", "cut_off", "several_faces"].includes(c))) runParts(faces);
        errors = 0;
      } catch (e) {
        errors += 1;
        console.warn("SkinIQ face check failed on a frame:", e);
        if (errors >= MAX_FRAME_ERRORS) fallback((e && e.message) || "detection error");
        return;
      } finally {
        busy = false;
      }
      if (inst.detector !== detect || !inst.timer || root.classList.contains("captured")) return;
      const [cls, text, codes] = verdict;
      inst.lastCodes = codes;
      passes = cls === "green" ? passes + 1 : 0;
      inst.aligned = passes >= PASS_FRAMES;
      if (cls === "green" && !inst.aligned) setState("red", "Hold still…");
      else setState(cls, text);
      shoot.disabled = !inst.aligned;
    }, 120);
  };

  const startCamera = async () => {
    root.classList.remove("captured");
    still.hidden = true; video.hidden = false; retake.hidden = true; shoot.hidden = false;
    shoot.disabled = true;
    setState(null, "Starting camera…");
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setState("red", "Camera needs a secure page (https or localhost). Upload a photo instead.");
      return;
    }
    try {
      if (!inst.stream) {
        inst.stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 960 } }, audio: false });
      }
      video.srcObject = inst.stream;
      await video.play();
    } catch (e) {
      setState("red", e && e.name === "NotAllowedError"
        ? "Camera access was blocked. Allow the camera for this site in your browser settings, or upload a photo."
        : "No camera found. Upload a photo instead.");
      return;
    }
    setState(null, "Loading face guide…");
    inst.detector = await loadDetector(data.static_base);
    if (!inst.stream) return;  // camera stopped (photo taken or page left) while loading
    if (inst.detector) startLoop();
    else fallback(detectorError || "the face detector could not load");
  };

  shoot.onclick = () => {
    if (shoot.disabled || video.readyState < 2) return;
    const [sx, sy, sw, sh] = visibleRegion(video);   // exactly what the guide checked
    const c = document.createElement("canvas");
    c.width = Math.round(sw); c.height = Math.round(sh);
    c.getContext("2d").drawImage(video, sx, sy, sw, sh, 0, 0, c.width, c.height);
    const url = c.toDataURL("image/jpeg", 0.92);
    stopCamera();
    still.src = url; still.hidden = false; video.hidden = true;
    root.classList.add("captured");
    setState(null, "Photo taken");
    shoot.hidden = true; retake.hidden = false;
    setStateValue("photo", url);
  };

  retake.onclick = () => { setStateValue("photo", null); startCamera(); };

  if (data && data.photo) {
    stopCamera();
    still.src = data.photo; still.hidden = false; video.hidden = true;
    root.classList.add("captured");
    setState(null, "Photo taken");
    shoot.hidden = true; retake.hidden = false;
  } else if (!inst.stream) {
    startCamera();
  }

  return () => stopCamera();
}
"""

_CAMERA = st.components.v2.component("skiniq_camera", html=_HTML, css=_CSS, js=_JS)


def face_camera(key: str = "face_camera", initial: bytes | None = None) -> tuple[bytes | None, bool]:
    """Show the guided camera. Returns (captured JPEG bytes or None, retaken).

    `initial` is a photo accepted earlier: it is shown as the captured still (with "Retake") when the
    component has no photo of its own yet. `retaken` is True once "Retake" was pressed, so the caller can
    drop the earlier photo; the live camera then starts again straight away.
    """
    state = st.session_state.get(key) or {}
    retaken = "photo" in state and state["photo"] is None
    current = state.get("photo")
    if not current and initial and not retaken:
        current = "data:image/jpeg;base64," + base64.b64encode(initial).decode()
    parts_models = None
    try:
        from app.services import browser_model_files, browser_models

        if browser_models():
            parts_models = browser_model_files()
    except Exception:  # models unavailable: the camera uses its rule-based checks
        parts_models = None
    result = _CAMERA(key=key, data={"photo": current, "static_base": STATIC_BASE, "settings": settings(),
                                    "parts_models": parts_models},
                     on_photo_change=lambda: None)
    url = result.photo or current  # state persists across reruns; None after "Retake"
    if not url or "," not in url:
        return None, retaken
    return base64.b64decode(url.split(",", 1)[1]), retaken
