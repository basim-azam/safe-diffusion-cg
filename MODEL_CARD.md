
# --- file: MODEL_CARD.md ---

## Classifier

This project uses a mid-block feature classifier trained on five classes with the following order:

```
['gore', 'hate', 'medical', 'safe', 'sexual']
```

`safe_idx=3` corresponds to the "safe" class.

**Weights:** `basimazam/safety-classifier-1280` (or local `models/safety_classifier_1280.pth`).

**Input features:** UNet mid-block features of the chosen SD pipeline.
