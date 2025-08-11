
# --- file: sdg/classifier/model.py ---

import torch
import torch.nn as nn
from typing import Optional

try:
    from huggingface_hub import hf_hub_download
    _HAS_HF = True
except Exception:
    _HAS_HF = False

CLASSIFIER_CLASS_NAMES = ['gore', 'hate', 'medical', 'safe', 'sexual']


class GlobalAvgPool(nn.Module):
    def forward(self, x):
        # x: (B, C, H, W)
        return x.mean(dim=[2, 3])


class Classifier1280(nn.Module):
    """Minimal classifier for mid-block features with 1280 channels."""
    def __init__(self, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            GlobalAvgPool(),              # (B,1280)
            nn.LayerNorm(1280),
            nn.Linear(1280, 512),
            nn.GELU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class Classifier320(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            GlobalAvgPool(),              # (B,320)
            nn.LayerNorm(320),
            nn.Linear(320, 256),
            nn.GELU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def get_classifier(sd_version: str, num_classes: int = 5) -> nn.Module:
    # For SD 1.4 / 1.5 / 2.1 mid-block channels are 1280 for base checkpoints
    # If your weights are 1280, prefer Classifier1280 regardless of variant
    return Classifier1280(num_classes=num_classes)


def load_classifier_weights(model: nn.Module, repo_or_path: Optional[str], filename: str = "safety_classifier_1280.pth"):
    """Load weights either from a local path or a Hugging Face repo id."""
    file_path = None
    if repo_or_path:
        if os.path.isdir(repo_or_path):
            file_path = os.path.join(repo_or_path, filename)
        else:
            # assume HF repo id
            if not _HAS_HF:
                raise RuntimeError("Install huggingface_hub to download weights from a repo id.")
            file_path = hf_hub_download(repo_or_path, filename)
    else:
        # default local models folder
        maybe = os.path.join(os.path.dirname(__file__), "..", "..", "models", filename)
        maybe = os.path.abspath(maybe)
        if os.path.exists(maybe):
            file_path = maybe
        else:
            raise FileNotFoundError(f"Classifier weights not found at {maybe}. Provide repo_or_path or place file under models/.")

    sd = torch.load(file_path, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    return model

