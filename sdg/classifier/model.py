# sdg/classifier/model.py
import torch
import torch.nn as nn
import logging

CLASSIFIER_CLASS_NAMES = ['gore', 'hate', 'medical', 'safe', 'sexual']

class AdaptiveClassifier(nn.Module):
    """
    Minimal CNN branch matching your training config:
    - Expects (B, 1280, 8, 8) features (mid-block tensor)
    - Outputs logits for 5 classes
    """
    def __init__(self, input_shape=(1280, 8, 8), num_classes=5):
        super().__init__()
        c, h, w = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(c, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Accept (B, C, H, W) or (C, H, W) and add batch if needed
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.features(x)
        return self.classifier(x)

def load_trained_classifier(checkpoint_path: str, device='auto'):
    """
    Load weights saved by your training script (adaptive_classifiers.py style).
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AdaptiveClassifier(input_shape=(1280, 8, 8), num_classes=5).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Try the common keys used in your project
    state = (
        ckpt.get('model_state_dict')
        or ckpt.get('state_dict')
        or ckpt  # raw state dict
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    logging.info(f"Loaded classifier: {checkpoint_path} on {device}")
    return model
