# Photo check: framing, visibility and lighting

`skiniq/vision/quality.py` decides whether a photo can be analysed. The same checks, with the same thresholds
(`quality.settings()`), run on uploaded photos, on captured photos (server), and live in the camera guide
(`app/components/camera.py`, in the browser), so the outline only turns green for a face the server will accept.

## Why trained models

YuNet (face detection) still finds a face when part of it is covered: on 900 dataset selfies with occluders drawn
on, its median score was 0.92 for clean faces, 0.90 with sunglasses and 0.85 with a mask. A first version with
hand-written colour and edge rules missed real glasses, sheet masks, hands and closed eyes in user testing, so the
visibility checks now use trained models:

| Model | What it does here | Source and licence | Size |
|---|---|---|---:|
| YuNet (OpenCV Zoo) | Face box and 5 landmarks; framing and pose | MIT | 0.2 MB |
| Face parsing, BiSeNet ResNet-18 (yakhyo/face-parsing via `jbrownkramer/face-parsing`) | Labels each pixel: skin, eyebrows, eyes, **glasses**, nose, lips, **hair**, **hat**… | MIT code; trained on CelebAMask-HQ (**non-commercial research and education only**) | 53 MB |
| MediaPipe palm detector + hand landmark model (OpenCV Zoo ONNX) | Hands touching or covering the face (face parsing labels hands as skin) | Apache-2.0 | 4 + 4 MB |
| CLIP ViT-B/32 image encoder (OpenAI; quantized ONNX by Xenova) + two small classifiers trained for SkinIQ (`skiniq/vision/assets/clip_probes.json`) | Drawing / cartoon vs real photo; hand on face (extra evidence) | MIT | 89 MB |

Models are downloaded on first use into `models/photo_check/` and checked against their SHA-256
(`skiniq/vision/face_models.py`). The camera runs YuNet, face parsing and the palm detector in the browser with
ONNX Runtime Web (CPU, no WebGL); the app copies the two larger models to `app/static/photo_check/` for it.
If the models cannot be downloaded, the server falls back to the earlier rule-based checks.

## Checks and messages

| Problem | How it is detected | Threshold (`face_models.PARTS`, `HANDS`, `quality.LIGHTING`) |
|---|---|---|
| Eye closed or covered | Visible eye pixels (white, iris) around each eye landmark, per eye | < 0.0015 × (eye distance)² |
| Glasses / sunglasses | Glasses pixels over **both** eyes (a phone or lens over one eye does not count); sunglasses if the lenses are dark | ≥ 15% of both eye areas; lens lightness < 0.55 × skin |
| Face mask over nose or mouth | Nose or lip pixels missing where the landmarks put them | nose < 10% or lips < 5% of their areas |
| Sheet / clay mask, colour filter | Median redness (a\*) of the skin: every real skin tone is reddish; white sheet masks, green clay and blue/grey filters are not | a\* < 3 |
| Forehead covered | Hair or hat pixels in the band just above the eyebrows (pure-black areas and areas outside the photo ignored) | > 55% |
| Hand on face | Palm detector on the photo and on a close-up of the face, each palm overlapping the face confirmed by the hand landmark model; plus the CLIP hand classifier | palm ≥ 0.8 or landmarks ≥ 0.5, or CLIP ≥ 0.7 |
| Drawing / cartoon | CLIP classifier | ≥ 0.5 |
| Black-and-white photo | 90th percentile HSV saturation of the photo | < 25 |
| Too dark | The brightest 5% of the photo (L\*) is dim (in a well-lit photo something is bright whatever the skin tone), or > 20% of skin pixels crushed to black | < 65 |
| Too bright / one side in shadow | Skin pixels blown out; cheek lightness differs | > 10%; > 30 L\* |
| Framing and pose: **captured photos** | Head and shoulders: face 30–60% of the photo height, chin above 80%, centred, not cut off; tilt ≤ 15°, not turned | `FRAMING` (same in the camera guide) |
| Framing and pose: **uploaded photos** | A close-up is fine (no neck or shoulders needed): face ≥ 30% of the photo height with room of 8% of the face size on every side, so the whole face is in the photo; centred (x 35–65%, y 25–65%); tilt ≤ 15°, not turned | `UPLOAD_FRAMING` |

Phone photos are rotated by their EXIF orientation before checking.

## Evaluation (real photos)

Labels were made by reviewing contact sheets of the 3,923 skin-type dataset selfies (by eye), plus 340 drawn faces
(huggan/selfie2anime anime, huggan/cartoon-faces, cgarciae/cartoonset; CC BY 4.0) and 100 real selfies from
huggan/selfie2anime. Photos were placed head and shoulders on a larger canvas, as a guided photo would be.

| Test | Result |
|---|---|
| 177 labelled selfies, 104 with a hand on the face (`training/results/photo_check/hand_labels.csv`) | 87% of hand photos rejected for the hand, 4% of the others wrongly |
| Drawings, each style held out of training | 72–100% caught at 0.5; real photos flagged 0.0–0.1%; YuNet finds no face in a further ~30% of drawings |
| Real problem photos (hands, glasses, sunglasses, closed eyes, fringe, cap) | All rejected with the right reason, on the server and live in the camera |
| Darkness | 4.2% of normal photos flagged (the dimmest, reviewed: dim lighting, no well-lit darker skin among them); 64% of the same photos at 45% exposure caught |
| 400 front-facing dataset selfies | 42% rejected; on review mostly genuine: hand on face 54, forehead covered 41, eyes closed/covered 39, uneven light 22, too dark 19, black-and-white 12, glasses 9, mask 9, unnatural skin colour 8 (this dataset is full of skincare poses) |

## Limitations

- The face-parsing weights are for non-commercial research and education (CelebAMask-HQ licence).
- Lightly tinted sunglasses can be reported as glasses (still rejected).
- Very wide smiles that nearly close the eyes can be reported as closed eyes.
- Labels were made by one reviewer from contact sheets; the hand set was partly chosen with the palm detector, so
  its recall may be optimistic.
- Skin-tone fairness of the darkness rule could not be measured directly (no skin-tone labels); it uses the
  brightest part of the photo rather than skin brightness for that reason.
