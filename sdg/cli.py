# sdg/cli.py
import typer, os
from pathlib import Path
from typing import Optional

import torch
from diffusers import StableDiffusionPipeline

from sdg.generation.cg_runner import SafeGuidanceRunner, CLASSIFIER_CLASS_NAMES
from sdg.classifier.model import load_trained_classifier

app = typer.Typer(add_completion=False, no_args_is_help=True)

@app.command("generate")
def generate(
    prompt: str = typer.Option(..., help="Text prompt"),
    out: Path = typer.Option(Path("out.png"), help="Output image path (.png)"),
    steps: int = typer.Option(30, help="Num inference steps"),
    scale: float = typer.Option(7.5, help="Classifier-free guidance scale"),
    negative_prompt: str = typer.Option("", help="Negative prompt"),
    seed: int = typer.Option(42, help="Random seed"),
    safety_scale: float = typer.Option(3.0, help="Safety guidance scale"),
    mid_fraction: float = typer.Option(0.5, help="Fraction of the schedule to apply CG (tail region)"),
    safe_idx: int = typer.Option(3, help="Index of 'safe' class"),
    sd_variant: str = typer.Option("runwayml/stable-diffusion-v1-5", help="Diffusers model id or local path"),
    checkpoint: Path = typer.Option(Path("models/safety_classifier_1280.pth"), help="Safety classifier .pth"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Device"),
    fp16: bool = typer.Option(True, help="Use fp16 autocast for UNet"),
):
    """
    Generate a single image using Classifier Guidance (CG) only.
    """
    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(sd_variant, safety_checker=None, requires_safety_checker=False)
    pipe = pipe.to(device)
    if fp16 and device.startswith("cuda"):
        # Keep UNet / VAE in half where applicable
        pipe.unet.to(dtype=torch.float16)
        if hasattr(pipe, "vae"):
            pipe.vae.to(dtype=torch.float16)

    # Load classifier (based on your adaptive_classifiers.py config)
    classifier = load_trained_classifier(str(checkpoint), device=device)

    runner = SafeGuidanceRunner(pipe=pipe, classifier=classifier)
    image = runner.generate_with_custom_cg(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=scale,
        negative_prompt=negative_prompt,
        seed=seed,
        safety_scale=safety_scale,
        mid_fraction=mid_fraction,
        safe_idx=safe_idx,
        fp16=fp16,
    )
    os.makedirs(Path(out).parent, exist_ok=True)
    image.save(out)
    typer.echo(f"Saved: {out}")

@app.command("classes")
def classes():
    """Print classifier class names and the index of 'safe'."""
    for i, n in enumerate(CLASSIFIER_CLASS_NAMES):
        mark = "<- safe" if n == "safe" else ""
        print(f"{i:2d}: {n} {mark}")

if __name__ == "__main__":
    app()
