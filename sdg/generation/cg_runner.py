
# --- file: sdg/generation/cg_runner.py ---

import os
import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional
from diffusers import StableDiffusionPipeline
from PIL import Image

from sdg.classifier.model import get_classifier, load_classifier_weights

CLASSIFIER_CLASS_NAMES = ['gore', 'hate', 'medical', 'safe', 'sexual']


@ dataclass
class CGParams:
    cg_scales: List[float] = None        # e.g., [3.5]
    mid_fracs: List[float] = None        # e.g., [0.5]
    safe_idx: int = 3                    # index of 'safe'


class SafeGuidanceRunner:
    def __init__(self, sd_version: str = "sd1_5", device: str = None, fp16: bool = True,
                 model_id: Optional[str] = None,
                 classifier_repo_or_path: Optional[str] = None,
                 classifier_filename: str = "safety_classifier_1280.pth"):
        self.sd_version = sd_version
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fp16 = fp16 and self.device.type == "cuda"

        # Choose model id
        if model_id is None:
            if sd_version == "sd1_4":
                model_id = "runwayml/stable-diffusion-v1-4"
            elif sd_version == "sd1_5":
                model_id = "runwayml/stable-diffusion-v1-5"
            elif sd_version == "sd2_1":
                model_id = "stabilityai/stable-diffusion-2-1-base"
            else:
                raise ValueError(f"Unsupported sd_version: {sd_version}")
        self.model_id = model_id

        dtype = torch.float16 if self.fp16 else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            safety_checker=None,
            feature_extractor=None
        ).to(self.device)
        self.pipe.enable_attention_slicing()

        # Classifier: architecture is chosen by sd_version (mid channels = 1280 for these models)
        self.classifier = get_classifier(sd_version=self.sd_version, num_classes=len(CLASSIFIER_CLASS_NAMES))
        load_classifier_weights(self.classifier, repo_or_path=classifier_repo_or_path, filename=classifier_filename)
        self.classifier.eval().to(self.device)

    @torch.no_grad()
    def _encode_prompt(self, prompt: str, negative_prompt: Optional[str] = ""):
        text_inp = self.pipe.tokenizer(prompt, padding="max_length", max_length=self.pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        neg_inp = self.pipe.tokenizer(negative_prompt or "", padding="max_length", max_length=self.pipe.tokenizer.model_max_length, return_tensors="pt")
        text_emb = self.pipe.text_encoder(text_inp.input_ids.to(self.device))[0].to(self.pipe.unet.dtype)
        uncond_emb = self.pipe.text_encoder(neg_inp.input_ids.to(self.device))[0].to(self.pipe.unet.dtype)
        return uncond_emb, text_emb

    def _capture_mid_features(self, latents: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor):
        """Forward through UNet with a hook on mid_block to collect features."""
        feats = {}
        handle = self.pipe.unet.mid_block.register_forward_hook(lambda m, i, o: feats.update({"mid": o[0] if isinstance(o, (list, tuple)) else o}))
        try:
            _ = self.pipe.unet(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states)
        finally:
            handle.remove()
        return feats.get("mid")

    def generate(self,
                 method: str,
                 prompt: str,
                 steps: int = 30,
                 scale: float = 7.5,
                 seed: Optional[int] = None,
                 out: Optional[str] = None,
                 negative_prompt: str = "",
                 cg_params: Optional[CGParams] = None,
                 image_size: int = 512) -> Image.Image:
        assert method == "cg", "Only Classifier Guidance (method='cg') is implemented"
        cg_params = cg_params or CGParams(cg_scales=[3.5], mid_fracs=[0.5], safe_idx=3)

        # Seeding
        generator = torch.Generator(device=self.device)
        if seed is None:
            seed = torch.seed() % (2**32 - 1)
        generator.manual_seed(int(seed))

        # Scheduler setup
        self.pipe.scheduler.set_timesteps(steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps

        # Latent init
        latents = torch.randn(
            (1, self.pipe.unet.config.in_channels, image_size, image_size),
            generator=generator, device=self.device, dtype=self.pipe.unet.dtype
        ) * self.pipe.scheduler.init_noise_sigma

        # Text encodings
        with torch.no_grad():
            uncond_emb, text_emb = self._encode_prompt(prompt, negative_prompt)
            text_embeddings = torch.cat([uncond_emb, text_emb], dim=0)

        # CG window (late steps)
        mid_frac = float(cg_params.mid_fracs[0])
        start_idx = int(steps * (1 - mid_frac))

        for i, t in enumerate(timesteps):
            # classifier-free guidance forward
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred_cfg = noise_uncond + scale * (noise_text - noise_uncond)

            # ---- CG (on-the-fly) ----
            if i >= start_idx:
                # Re-forward UNet with UNCOND embeddings to collect mid features at this step
                with torch.enable_grad():
                    latents.requires_grad_(True)
                    mid_latent_in = self.pipe.scheduler.scale_model_input(latents, t)
                    mid_feats = self._capture_mid_features(mid_latent_in, t, encoder_hidden_states=uncond_emb)

                    # Classifier expects shape (B, C, H, W); global avg pool -> logits
                    feats = mid_feats
                    logits = self.classifier(feats)
                    probs = F.softmax(logits[0], dim=-1)
                    safe_prob = probs[cg_params.safe_idx]
                    loss = -torch.log(safe_prob + 1e-6)
                    grad = torch.autograd.grad(loss, latents, retain_graph=False)[0]

                cg_scale = float(cg_params.cg_scales[0])
                latents = (latents.detach() - cg_scale * grad.to(latents.dtype)).to(latents.dtype)
                del grad, feats, logits, probs, mid_feats
                torch.cuda.empty_cache()

            # scheduler step
            latents = self.pipe.scheduler.step(noise_pred_cfg, t, latents).prev_sample

        # Decode latents to image
        with torch.no_grad():
            latents = latents / self.pipe.vae.config.scaling_factor
            image = self.pipe.vae.decode(latents.to(self.pipe.vae.dtype)).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
            pil = Image.fromarray((image * 255).round().astype("uint8"))

        if out:
            os.makedirs(out, exist_ok=True)
            pil.save(os.path.join(out, "cg_output.png"))
        return pil