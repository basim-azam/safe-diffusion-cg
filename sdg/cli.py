
# --- file: sdg/cli.py ---

import argparse
from sdg.generation.cg_runner import SafeGuidanceRunner, CGParams


def build_parser():
    p = argparse.ArgumentParser(prog="sdg", description="Safe Diffusion Guidance (Classifier Guidance)")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate an image with Classifier Guidance")
    g.add_argument("--prompt", type=str, required=True)
    g.add_argument("--method", type=str, default="cg")
    g.add_argument("--safe-idx", type=int, default=3)
    g.add_argument("--cg-scales", type=float, nargs="+", default=[3.5])
    g.add_argument("--mid-fracs", type=float, nargs="+", default=[0.5])
    g.add_argument("--steps", type=int, default=30)
    g.add_argument("--scale", type=float, default=7.5)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--sd-version", type=str, default="sd1_5", choices=["sd1_4", "sd1_5", "sd2_1"])
    g.add_argument("--model-id", type=str, default=None)
    g.add_argument("--classifier", type=str, default=None, help="HF repo id or local dir")
    g.add_argument("--classifier-file", type=str, default="safety_classifier_1280.pth")
    g.add_argument("--out", type=str, default="out/")

    return p


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)

    if args.cmd == "generate":
        runner = SafeGuidanceRunner(
            sd_version=args.sd_version,
            model_id=args.model_id,
            classifier_repo_or_path=args.classifier,
            classifier_filename=args.classifier_file,
        )
        img = runner.generate(
            method=args.method,
            prompt=args.prompt,
            steps=args.steps,
            scale=args.scale,
            seed=args.seed,
            out=args.out,
            cg_params=CGParams(cg_scales=args.cg_scales, mid_fracs=args.mid_fracs, safe_idx=args.safe_idx),
        )
        print(f"Saved: {args.out}/cg_output.png")


if __name__ == "__main__":
    main()
