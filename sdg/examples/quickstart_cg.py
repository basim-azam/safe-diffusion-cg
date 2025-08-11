# examples/quickstart_cg.py
import torch
from diffusers import StableDiffusionPipeline
from sdg.generation.cg_runner import SafeGuidanceRunner, CLASSIFIER_CLASS_NAMES
from sdg.classifier.model import load_trained_classifier

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)

    # Optional: fp16 on CUDA
    if device == "cuda":
        pipe.unet.to(dtype=torch.float16)
        pipe.vae.to(dtype=torch.float16)

    classifier = load_trained_classifier("models/safety_classifier_1280.pth", device=device)

    runner = SafeGuidanceRunner(pipe=pipe, classifier=classifier)
    img = runner.generate_with_custom_cg(
        prompt="a family having dinner",
        num_inference_steps=30,
        guidance_scale=7.5,
        negative_prompt="nsfw, nude, gore, sexual content, hateful content",
        seed=42,
        safety_scale=3.0,
        mid_fraction=0.5,
        safe_idx=CLASSIFIER_CLASS_NAMES.index("safe"),
        fp16=True,
    )
    img.save("out/quickstart.png")
    print("Saved to out/quickstart.png")

if __name__ == "__main__":
    main()
