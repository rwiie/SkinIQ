# Multi-label annotation guidelines

Each photo gets **six independent yes/no labels**, one per skin concern. A photo can have
any number of concerns, including none (clear skin). Use the same rules for every photo, so
the labels are consistent.

## General rules

1. **Tick a concern if it is clearly visible anywhere on the skin in the photo.** Severity
   doesn't matter: one clear lesion is enough.
2. **Don't tick what you can't see clearly.** If you'd have to guess, leave it unticked.
3. **Close-ups start with their dataset folder's concern ticked, and Roboflow photos with
   the concerns their annotators boxed.** Keep a ticked concern if you can see it; untick it
   only if it's clearly not there. Then add any other concerns you see: the Roboflow
   annotators often labelled only one or two concerns per photo.
4. **Tick "Unsure"** when the photo is too blurry, small, dark, filtered or heavily made-up
   to judge some concern. The labels are still saved; unsure photos can be left out of the
   test set.
5. **Press "Unusable"** for photos with no real human skin to judge (cartoons, product
   shots, mostly text, a hand only). These are excluded.
6. Aim for about 5–10 seconds per photo. Don't revisit old photos to "fix" them unless you
   made a clear mistake (use ◀ Back).

## Concern definitions

| Concern | Tick when you see | Don't tick for |
|---|---|---|
| **Acne** | Active inflamed bumps: red papules, pustules (white or yellow head), nodules or cysts; whiteheads (closed comedones) | Dark or red *marks* left after acne has healed (→ hyperpigmentation or redness); moles; insect bites |
| **Blackheads** | Small dark or black dots plugging pores (open comedones), usually on the nose, chin or forehead | Freckles; stubble; dark spots that are flat and wider than a pore |
| **Hyperpigmentation** | Flat areas darker than the surrounding skin: dark spots, sun spots, melasma patches, post-acne dark marks, dense freckling | A single mole; shadows; natural under-eye darkness |
| **Redness** | Diffuse red or flushed areas (cheeks, nose, chin), rosacea, visible irritation or broken capillaries | Red acne bumps alone (→ acne); blush or lipstick; a red colour cast over the whole photo from lighting |
| **Enlarged pores** | Clearly visible open pores or "orange-peel" texture, usually on the cheeks or nose | Pores you can only see by zooming into a high-resolution photo; blackheads alone (→ blackheads) |
| **Wrinkles** | Any visible lines or folds, **including fine lines**: forehead lines, crow's feet, under-eye lines, lines around the mouth, nasolabial folds, crepey texture | Smooth skin with a temporary crease from a big smile or squint only |

**Clear skin:** leave all six unticked and press **Save & next**.

## Where labels go

The Colab notebook `training/annotate_colab.ipynb` saves every answer immediately to
`My Drive/SkinIQ/labels/multilabel_labels.csv`. Copies of the same original photo share
its labels automatically.

## Report notes

- Labels were created by one annotator using these written criteria. If possible, ask a
  second person (e.g. your supervisor) to label the same 50 test photos; the agreement
  between you (Cohen's kappa per concern) shows how reliable the labels are.
- Annotating the test set first, from a stratified sample, keeps the evaluation independent
  of the training data.
