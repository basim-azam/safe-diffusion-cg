
# --- file: examples/quickstart_cg.py ---

from sdg.generation.cg_runner import SafeGuidanceRunner, CGParams

if __name__ == "__main__":
    runner = SafeGuidanceRunner(sd_version="sd1_5", classifier_repo_or_path="basimazam/safety-classifier-1280")
    img = runner.generate(
        method="cg",
        prompt="a family having dinner",
        steps=30,
        scale=7.5,
        seed=42,
        cg_params=CGParams(cg_scales=[3.5], mid_fracs=[0.5], safe_idx=3),
        out="out/",
    )
    print("Wrote out/cg_output.png")
