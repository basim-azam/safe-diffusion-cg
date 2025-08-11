# sdg/cli.py
import typer, os
from pathlib import Path
from typing import Optional, List

import torch
from diffusers import StableDiffusionPipeline

from sdg.generation.cg_runner import SafeGuidanceRunner, CLASSIFIER_CLASS_NAMES
from sdg.classifier.model import load_trained_classifier

app = typer.Typer(add_completion=False, no_args_is_help=True)

def _parse_floats(s: str) -> list[float]:
    try:
        return [float(x.strip()) for x in s.split(",") if x.strip()]
    except Exception:
        raise typer.BadParameter("Provide a float or comma-separated floats, e.g. '3.5' or '2.5,3.5'")

@app.command("generate")
def generate(
    prompt: str = typer.Option(..., help="Text prompt"),
    method: str = typer.Option("cg", "--method", help="Guidance method (only 'cg' is supported)"),
    out: Path = typer.Option(Path("out/out.png"), help="Output image path (.png) or directory if --grid"),
    steps: int = typer.Option(30, "--steps", help="Num inference steps"),
    scale: float = typer.Option(7.5, "--scale", help="Classifier-free guidance scale"),
    negative_prompt: str = typer.Option("", help="Negative prompt"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    cg_scales: str = typer.Option("3.0", "--cg-scales", help="CG strength(s). Float or comma-separated list."),
    mid_fracs: str = typer.Option("0.5", "--mid-fracs", help="Tail fraction(s). Float or comma-separated list."),
    safe_idx: int = typer.Option(3, "--safe-idx", help="Index of 'safe' class (see `sdg classes`)"),
    sd_variant: str = typer.Option("runwayml/stable-diffusion-v1-5", help="Diffusers model id or local path"),
    checkpoint: Path = typer.Option(Path("models/safety_classifier_1280.pth"), help="Safety classifier .pth"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Device"),
    fp16: bool = typer.Option(True, help="Use fp16 autocast for UNet"),
    grid: bool = typer.Option(False, help="If multiple scales/fracs provided, save a grid to OUT instead of separate files."),
):
    """Generate image(s) using **Classifier Guidance (CG)**.

    - If multiple --cg-scales and/or --mid-fracs are provided, a sweep is run.
    - Only CG method is implemented.
    """
    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(sd_variant, safety_checker=None, requires_safety_checker=False)
    pipe = pipe.to(device)
    if fp16 and device.startswith("cuda"):
        pipe.unet.to(dtype=torch.float16)
        if hasattr(pipe, "vae"):
            pipe.vae.to(dtype=torch.float16)

    # Load classifier
    classifier = load_trained_classifier(str(checkpoint), device=device)

    # Parse sweep values
    scales = _parse_floats(cg_scales)
    fracs  = _parse_floats(mid_fracs)

    runner = SafeGuidanceRunner(pipe=pipe, classifier=classifier)

    # Ensure output dir
    out = Path(out)
    if grid or len(scales) > 1 or len(fracs) > 1:
        out_dir = out if out.suffix.lower() != ".png" else out.parent
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)

    seeds = [seed]
    images = []
    labels = []

    for s in scales:
        for f in fracs:
            img = runner.generate_with_custom_cg(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=scale,
                negative_prompt=negative_prompt,
                seed=seed,
                safety_scale=float(s),
                mid_fraction=float(f),
                safe_idx=safe_idx,
                fp16=fp16,
            )
            images.append(img)
            labels.append(f"cg={s}, frac={f}")

            # Save single
            if not (grid or len(scales) > 1 or len(fracs) > 1):
                img.save(out)
                typer.echo(f"Saved: {out}")

            seed += 1  # nudge seed per sweep item

    # Optional grid save
    if grid or len(images) > 1:
        try:
            from PIL import Image, ImageDraw, ImageFont
            cols = len(fracs)
            rows = len(scales)
            w, h = images[0].size
            grid_img = Image.new("RGB", (cols*w, rows*h), (0,0,0))
            for i, img in enumerate(images):
                r = i // cols
                c = i % cols
                grid_img.paste(img, (c*w, r*h))
            grid_path = out / "grid.png" if out.is_dir() else out.with_suffix("").with_name(out.stem + "_grid.png")
            grid_img.save(grid_path)
            typer.echo(f"Saved grid: {grid_path}")
        except Exception as e:
            typer.echo(f"Grid save failed: {e}")

@app.command("classes")
def classes():
    """Print classifier class names and the index of 'safe'."""
    for i, n in enumerate(CLASSIFIER_CLASS_NAMES):
        mark = "<- safe" if n == "safe" else ""
        print(f"{i:2d}: {n} {mark}")

if __name__ == "__main__":
    app()
