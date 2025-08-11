
# --- file: SAFETY_GUIDE.md ---

## What CG does
Classifier Guidance (CG) nudges the latent towards higher probability of the **safe** class as determined by a lightweight classifier applied to the UNet **mid-block** features. At late diffusion steps (controlled by `mid_frac`), we compute a loss `-log p_safe` and apply its gradient to the latent before the scheduler update.

## Recommended defaults
- `safe_idx = 3`
- `cg_scale ≈ 3.5`
- `mid_frac ≈ 0.5`
- `steps = 30`, `scale = 7.5`

## Known failure cases
- Domain shift or non-512×512 configurations can reduce effectiveness.
- Extremely adversarial prompts may overpower the gentle CG signal.
- Very low CFG (`--scale`) may degrade image quality regardless of CG.
- Mixed precision instability if tensors have mismatched dtypes; this repo follows the pipeline dtype and casts correctly.

## Limitations
CG is a **soft guidance**: it improves safety likelihood but is **not** a hard content filter. Consider combining with post-hoc moderation in downstream applications.

## Reproducibility
- Pin model ids and seeds in docs/CI.
- Record CLI used to produce examples.

