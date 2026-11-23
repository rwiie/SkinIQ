# Bundled MediaPipe face detector (browser)

Used by the camera guide (`app/components/camera.py`) for live face detection in the browser.

| File | Source | Licence |
|---|---|---|
| `vision_bundle.mjs`, `wasm/vision_wasm_internal.{js,wasm}` | npm `@mediapipe/tasks-vision@0.10.14` | Apache-2.0 |
| `blaze_face_short_range.tflite` | Google MediaPipe model (`storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1`) | Apache-2.0 |

Served by Streamlit static file serving (`server.enableStaticServing = true`) at `app/static/mediapipe/`.
Bundled (not loaded from a CDN) so the camera guide works offline and behind restrictive networks.
