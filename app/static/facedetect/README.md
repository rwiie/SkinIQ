# Bundled browser face detector

Used by the camera guide (`app/components/camera.py`) for live face detection in the browser.
It runs on the CPU (WebAssembly), so it works when WebGL / graphics acceleration is unavailable,
and it is the same detector as the server-side photo check (`skiniq/vision/quality.py`).

| File | Source | Licence |
|---|---|---|
| `ort.wasm.bundle.min.mjs`, `ort-wasm-simd-threaded.{mjs,wasm}` | npm `onnxruntime-web@1.22.0` (CPU/WebAssembly build) | MIT |
| `yunet_2023mar_dynamic.onnx` | OpenCV Zoo YuNet (`face_detection_yunet_2023mar.onnx`, same file as `skiniq/vision/assets/`) with the input height/width made dynamic (graph unchanged) | MIT |

Served by Streamlit static file serving (`server.enableStaticServing = true`) at `app/static/facedetect/`.
Bundled (not loaded from a CDN) so the camera guide works offline and behind restrictive networks.
