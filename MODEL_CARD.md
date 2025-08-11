# Safety Classifier (mid-block, 1280×8×8)

**Intended use:** steer Stable Diffusion generations *during sampling* toward safer content by maximizing `p(safe)` from a mid-UNet classifier.

- **Architecture:** small CNN head over mid-block features (C=1280, H=W=8), 5-way logits: `['gore','hate','medical','safe','sexual']`.
- **Checkpoint:** `models/safety_classifier_1280.pth` (inference-only).
- **Integration:** Used only in CG; not for post-hoc filtering.

## Supported base models

- Stable Diffusion `v1-4`, `v1-5`, `v2-1` via 🤗 Diffusers.

## Limitations

- The classifier’s notion of “safe” is dataset- and training-protocol-dependent.
- Domain shifts (art styles, languages, composites) can reduce reliability.
- CG competes with image fidelity at very high `--cg-scales` or very long `--mid-fracs`.

## Recommended defaults

- `--cg-scales 3.0–4.0`, `--mid-fracs 0.4–0.6`, `--safe-idx 3` (safe).
- Steps: 25–35; CFG scale: 7.0–8.5.
