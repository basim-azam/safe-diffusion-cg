
# --- file: README.md ---

# Safe Diffusion Guidance (Minimal CG)

Minimal, production-ready repository to run **Classifier Guidance (CG)** during Stable Diffusion generation using a mid-block feature classifier.

- Supports **SD 1.4 / 1.5 / 2.1**
- Works with your public weights: **`basimazam/safety-classifier-1280`**
- Exposes a simple CLI: `sdg generate ...`

## Install

```bash
pip install -e .
```

## Weights

- Easiest: let the package auto-download from Hugging Face by passing `--classifier basimazam/safety-classifier-1280`.
- Or, drop `safety_classifier_1280.pth` under `models/`.

## Quickstart

```bash
# SD 1.5 + CG with defaults
sdg generate \
  --prompt "a family having dinner" \
  --method cg \
  --sd-version sd1_5 \
  --classifier basimazam/safety-classifier-1280 \
  --cg-scales 3.5 \
  --mid-fracs 0.5 \
  --safe-idx 3 \
  --steps 30 \
  --scale 7.5 \
  --seed 42 \
  --out out/
```

## Programmatic

```python
from sdg.generation.cg_runner import SafeGuidanceRunner, CGParams
runner = SafeGuidanceRunner(sd_version="sd1_5", classifier_repo_or_path="basimazam/safety-classifier-1280")
img = runner.generate(
    method="cg", prompt="a cute cat", steps=30, scale=7.5, seed=1234,
    cg_params=CGParams(cg_scales=[3.5], mid_fracs=[0.5], safe_idx=3), out="out/")
```

## Supported SD versions
- `sd1_4` → `runwayml/stable-diffusion-v1-4`
- `sd1_5` → `runwayml/stable-diffusion-v1-5`
- `sd2_1` → `stabilityai/stable-diffusion-2-1-base`

## Notes
- Only **`generate(method="cg")`** is implemented. No post-hoc classifiers, no ESD/UCE/SLD.
- FP16 is auto-enabled on CUDA devices.
