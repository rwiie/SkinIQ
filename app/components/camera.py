"""Camera with a head-and-shoulders guide and live face detection (in the browser).

The outline is red until exactly one face is centred in it at the right distance and in
enough light, then turns green and the "Take photo" button is enabled. Face detection
runs in the browser with MediaPipe (bundled in app/static/mediapipe, served by
Streamlit static serving). If the detector cannot load, the guide stays grey, the photo
can still be taken, and the server-side quality check (skiniq.vision.quality) decides.

Result, kept in st.session_state[key]["photo"]: a JPEG data URL, or None after "Retake".
"""

from __future__ import annotations

import base64

import streamlit as st

STATIC_BASE = "app/static/mediapipe/"

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

// Guide geometry in the 640x480 guide space (matches the SVG ellipse).
const OVAL = { cx: 320, cy: 205, rx: 118, ry: 152 };

let detectorPromise = null;
function loadDetector(base) {
  if (!detectorPromise) {
    detectorPromise = (async () => {
      const root = new URL(base, document.baseURI).href;
      const vision = await import(root + "vision_bundle.mjs");
      const files = await vision.FilesetResolver.forVisionTasks(root + "wasm");
      return vision.FaceDetector.createFromOptions(files, {
        baseOptions: { modelAssetPath: root + "blaze_face_short_range.tflite" },
        runningMode: "VIDEO",
        minDetectionConfidence: 0.6,
      });
    })().catch((e) => { console.warn("SkinIQ face detector unavailable:", e); detectorPromise = null; return null; });
  }
  return detectorPromise;
}

// Map a box from video pixels into the 640x480 guide space (object-fit: cover, mirrored).
function toGuide(box, vw, vh) {
  const scale = Math.max(640 / vw, 480 / vh);
  const ox = (640 - vw * scale) / 2, oy = (480 - vh * scale) / 2;
  const cx = (box.originX + box.width / 2) * scale + ox;
  return { cx: 640 - cx, cy: (box.originY + box.height / 2) * scale + oy, w: box.width * scale, h: box.height * scale };
}

function meanLuma(video, box, canvas) {
  const s = 32;
  canvas.width = s; canvas.height = s;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(video, box.originX, box.originY, box.width, box.height, 0, 0, s, s);
  const px = ctx.getImageData(0, 0, s, s).data;
  let sum = 0;
  for (let i = 0; i < px.length; i += 4) sum += 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2];
  return sum / (s * s);
}

function judge(faces, video, canvas) {
  if (faces.length === 0) return ["red", "No face detected — look straight at the camera"];
  if (faces.length > 1) return ["red", "More than one face — only you should be in the frame"];
  const box = faces[0].boundingBox;
  const g = toGuide(box, video.videoWidth, video.videoHeight);
  const ovalW = OVAL.rx * 2;
  if (Math.abs(g.cx - OVAL.cx) > OVAL.rx * 0.35 || Math.abs(g.cy - OVAL.cy) > OVAL.ry * 0.35)
    return ["red", "Centre your face inside the outline"];
  if (g.w < ovalW * 0.5) return ["red", "Move closer"];
  if (g.w > ovalW * 1.15) return ["red", "Move back a little"];
  const luma = meanLuma(video, box, canvas);
  if (luma < 60) return ["red", "Too dark — face a window or a light"];
  if (luma > 215) return ["red", "Too bright — avoid direct light on your face"];
  return ["green", "Face detected — hold still and take the photo"];
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

  const startLoop = () => {
    stopLoop();
    const canvas = document.createElement("canvas");
    inst.timer = setInterval(() => {
      if (!inst.detector || video.readyState < 2) return;
      let faces = [];
      try { faces = inst.detector.detectForVideo(video, performance.now()).detections || []; } catch (e) { return; }
      const [cls, text] = judge(faces, video, canvas);
      inst.aligned = cls === "green";
      setState(cls, text);
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
    if (inst.detector) {
      startLoop();
    } else {
      setState(null, "Face guide unavailable — fit your face in the outline; the photo will be checked after you take it");
      shoot.disabled = false;
    }
  };

  shoot.onclick = () => {
    if (shoot.disabled || video.readyState < 2) return;
    const c = document.createElement("canvas");
    c.width = video.videoWidth; c.height = video.videoHeight;
    c.getContext("2d").drawImage(video, 0, 0);
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


def face_camera(key: str = "face_camera") -> bytes | None:
    """Show the guided camera; return the captured JPEG bytes, or None."""
    current = (st.session_state.get(key) or {}).get("photo")
    result = _CAMERA(key=key, data={"photo": current, "static_base": STATIC_BASE},
                     on_photo_change=lambda: None)
    url = result.photo  # state persists across reruns; None before a capture and after "Retake"
    if not url or "," not in url:
        return None
    return base64.b64decode(url.split(",", 1)[1])
