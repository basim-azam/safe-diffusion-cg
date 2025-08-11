# Safe Diffusion Guidance (Classifier Guidance)

Minimal repo to run **Classifier Guidance (CG)** during Stable Diffusion generation using a mid-block safety classifier.

> This package implements only the on-the-fly CG path (no post-hoc classification, no ESD/UCE/SLD, no dataset utilities).

## Install

```bash
pip install -e .
```

Or run from source after installing deps:

```bash
pip install -r requirements.txt
```

## Models

Place `safety_classifier_1280.pth` in `models/`. See `models/README.md`.

Supported base models (tested): **SD v1.4, v1.5, v2.1**.

## CLI

```bash
sdg generate \
  --prompt "a family having dinner" \
  --method cg  # implied; only CG is implemented
  --cg-scales 3.5 \
  --mid-fracs 0.5 \
  --safe-idx 3 \
  --steps 30 \
  --scale 7.5 \
  --seed 42 \
  --out out/
```

- `--cg-scales` and `--mid-fracs` accept a float **or** comma-separated list. When multiple values are provided, a sweep is run and (optionally) a grid is saved.

List classes:
```bash
sdg classes
```

## Python quickstart

See [`examples/quickstart_cg.py`](examples/quickstart_cg.py).

## How it works (high level)

- We register a mid-block forward hook on the UNet, capture the 1280×8×8 tensor.
- A lightweight CNN classifier predicts logits over categories (`['gore','hate','medical','safe','sexual']`). We treat the **`safe`** index (default 3) as the target.
- During a chosen tail window of the schedule (`mid_fraction`), we backpropagate `-log p(safe)` to the latents and **take a step** in the direction that increases `p(safe)` by `safety_scale`.
- This is applied alongside normal classifier-free guidance (CFG).

## Notes

- FP16 is supported for UNet/VAE. For the classifier, we evaluate in its native dtype (typically fp32).
- Scheduler/device handling follows Diffusers execution device rules and works with CPU offload.
- Only the CG method is included; no safety checker.

See **SAFETY_GUIDE.md** and **MODEL_CARD.md** for details.
