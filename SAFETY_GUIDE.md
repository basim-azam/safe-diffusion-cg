# Safety & Ethics Guide

**Purpose:** Provide an *on-the-fly* safety steering signal during generation (Classifier Guidance, CG). This repo does **not** implement post-hoc filtering or removal techniques.

## What CG does

- Applies a gradient step to the latents that **increases the classifier’s probability of the `safe` class** during a chosen tail window of the sampling schedule.
- Works alongside standard classifier-free guidance (CFG).

## Practical guidance

- Start with: `--cg-scales 3.5`, `--mid-fracs 0.5`, `--safe-idx 3`.
- If unsafe hints remain, try raising `--cg-scales` or increasing `--mid-fracs` (e.g., `0.6`). If images look washed out or off-topic, lower them.
- Keep CFG scale reasonable (6.5–8.5). Extremely high CFG and CG can fight each other.

## Known failure cases

- **Ambiguous prompts:** prompts that implicitly suggest sensitive themes may require higher CG or explicit negative prompts.
- **Style loopholes:** stylized or abstract content can evade classifier signal.
- **Distribution shift:** content outside training domains may yield unreliable `safe` predictions.

## Limitations & disclaimers

- No safety system is perfect. Manual review is still advised for high-stakes use.
- This repo intentionally excludes post-hoc filtering, ESD/UCE/SLD, and dataset construction utilities.
