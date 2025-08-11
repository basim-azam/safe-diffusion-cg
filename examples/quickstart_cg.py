from diffusers import StableDiffusionPipeline
from sdg.generation.cg_runner import SafeGuidanceRunner
from sdg.classifier.model import load_trained_classifier

def main():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        safety_checker=None, requires_safety_checker=False
    )
    pipe = pipe.to("cuda")
    pipe.unet.to(dtype=pipe.unet.dtype)

    classifier = load_trained_classifier("models/safety_classifier_1280.pth", device="cuda")
    runner = SafeGuidanceRunner(pipe, classifier)

    img = runner.generate_with_custom_cg(
        prompt="a family having dinner",
        num_inference_steps=30,
        guidance_scale=7.5,
        negative_prompt="",
        seed=42,
        safety_scale=3.5,
        mid_fraction=0.5,
        safe_idx=3,
        fp16=True,
    )
    img.save("out/quickstart.png")
    print("Saved: out/quickstart.png")

if __name__ == "__main__":
    main()
