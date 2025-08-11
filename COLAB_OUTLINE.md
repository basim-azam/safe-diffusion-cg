# Colab Demo Outline

1. `pip install diffusers transformers accelerate safetensors Pillow tqdm`
2. `git clone https://github.com/<you>/safe-diffusion-guidance && cd safe-diffusion-guidance`
3. Place or `wget` the classifier to `models/safety_classifier_1280.pth`
4. `pip install -e .`
5. Run:
```python
from sdg.generation.cg_runner import SafeGuidanceRunner
from sdg.classifier.model import load_trained_classifier
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None, requires_safety_checker=False).to("cuda")
classifier = load_trained_classifier("models/safety_classifier_1280.pth", device="cuda")
runner = SafeGuidanceRunner(pipe, classifier)
img = runner.generate_with_custom_cg(prompt="a family having dinner", num_inference_steps=30, guidance_scale=7.5, seed=42, safety_scale=3.5, mid_fraction=0.5, safe_idx=3, fp16=True)
img.save("out.png")
```
