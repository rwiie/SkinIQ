# Photo check: framing, visibility and lighting

`skiniq/vision/quality.py` decides whether a photo can be analysed. The same checks, with the same
thresholds (`quality.settings()`), run on uploaded photos, on captured photos, and live in the camera guide
(`app/components/camera.py`), so the outline only turns green for a photo the server will accept.

## Why the face detector alone is not enough

YuNet finds faces even when they are partly covered. On 900 dataset selfies with occluders drawn on, its median
detection score stayed high: 0.92 for clean faces, 0.90 with sunglasses, 0.85 with a mask, 0.91 with one eye
covered. So "a face was detected" does not mean the face is fully visible.

## How each part of the face is checked

YuNet gives five landmarks (eyes, nose tip, mouth corners). The face is warped so the eyes and the mouth centre
land on fixed points (192 × 224 px), then small patches (eyes, forehead, cheeks, nose, mouth, chin, nose
bridge, under the eyes) are measured. Every measure is relative to the person's **own** skin (cheeks and nose),
so it works the same for every skin tone.

| Problem | Signal | Threshold |
|---|---|---|
| Sunglasses | Both eyes dark (mean lightness < 0.35 × skin) and mostly dark pixels (> 80%) | `sunglasses_*` |
| Glasses | Strong edges across the nose bridge (> 2.8 × cheek detail) and along the lower rims (> 2.2 ×) | `glasses_*` |
| Eye covered (hand, hair, object) | An eye with almost no contrast (< 0.06 × skin) that is not dark | `eye_*` |
| Face mask | Nose/mouth/chin colour far from the forehead (> 18) with no red lips, or a smooth mouth area (< 0.035) with no red lips | `mask_*` |
| One side covered | Cheek colours differ (> 20), or one half has much less detail than the other (< 0.5) | `half_*` |
| Too dark / too bright | > 20% of skin pixels crushed to black / > 10% blown out to white | `LIGHTING` |
| One side in shadow | Cheek lightness differs by > 30 (L*, 0–100) | `max_side_diff` |
| Framing | Face box 30–60% of the photo height, centre within the middle band, chin in the upper 80%, not touching the edges | `FRAMING` |
| Pose | Head tilted > 15°, or nose off the eye midline by > 0.25 × eye distance (turned) | `max_roll_deg`, `max_yaw` |

Lighting is judged on clipped and crushed skin, not on average brightness: an average-brightness rule would
reject darker skin in good light.

## Calibration

Data: 900 selfies from the skin-type dataset (one face, detection score ≥ 0.85), and copies of each with
sunglasses, glasses, a face mask, a dark object over one half, a skin-coloured "hand" over one half, or a hand
over one eye drawn on. For the end-to-end test each photo was placed head and shoulders on a larger canvas, as a
guided photo would be (600 photos).

| Photos | Rejected | Main reason given |
|---|---:|---|
| With sunglasses | 93% | sunglasses (78%) |
| With glasses | 90% | glasses (78%) |
| With a face mask | 100% | mask (85%) |
| Dark object over one half | 98% | one side in shadow / covered |
| Skin-coloured hand over one half | 95% | eye covered (48%), no face (27%) |
| Hand over one eye | 99% | eye covered (95%) |
| Clean, facing the camera (400) | 20% | too dark 6%, uneven light 6%, half covered 4% |
| Clean, all poses | 39% | turned 15%, tilted 14% (posed beauty photos) |

Most "clean" rejections are real problems: reviewing them showed censor bars and cut-out boxes over faces,
hair across the face, phones, face packs, real glasses and sunglasses, strong side light and profile poses.

**Fairness across skin tones** (clean photos split into thirds by skin lightness: darker / medium / lighter):
too dark 4.9% / 6.2% / 7.1%, one side in shadow 9.1% / 7.9% / 7.7% (at the calibration threshold of 25; the app
uses 30), too bright 0% / 0% / 3.5% (lighter skin really does blow out first).

## Limitations

- Rule-based, not a trained occlusion model. Occluders in the calibration were drawn on, so real objects with
  texture (a patterned mask, a textured hand) may be caught less often, and some real photos are rejected with
  a less precise reason (for example black-framed glasses in a black-and-white photo reported as sunglasses).
- Thin rimless glasses and a beard can be confused with no glasses and a mask respectively.
- The calibration photos are from a public dataset, not users of the app; the guided camera should give cleaner
  photos than these.
