# sdg/generation/cg_runner.py
import torch
from PIL import Image

CLASSIFIER_CLASS_NAMES = ['gore', 'hate', 'medical', 'safe', 'sexual']

@torch.no_grad()
def _encode_text(pipe, prompt, negative_prompt, dtype):
    device = pipe.device
    tok = pipe.tokenizer
    te  = pipe.text_encoder

    text_ids = tok(
        prompt,
        padding="max_length",
        max_length=tok.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    uncond_ids = tok(
        negative_prompt,
        padding="max_length",
        max_length=tok.model_max_length,
        return_tensors="pt",
    ).input_ids.to(device)

    text_emb   = te(text_ids)[0].to(dtype)
    uncond_emb = te(uncond_ids)[0].to(dtype)
    return torch.cat([uncond_emb, text_emb], dim=0), uncond_emb


class SafeGuidanceRunner:
    """
    Minimal runner that exposes CG-only generation, preserving your current behavior.
    """

    def __init__(self, pipe, classifier):
        self.pipe = pipe
        self.classifier = classifier

    def generate_with_custom_cg(
        self,
        prompt: str,
        *,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        negative_prompt: str = "",
        seed: int = 0,
        safety_scale: float = 3.0,
        mid_fraction: float = 0.5,
        safe_idx: int = 3,
        fp16: bool = True,
    ):
        device = self.pipe.device
        pipe_dtype = self.pipe.unet.dtype
        gen = torch.Generator(device).manual_seed(int(seed))

        # Classifier device / mode
        classifier = self.classifier.to(device)
        classifier.eval()

        # Init latents
        latents = torch.randn(
            (1, self.pipe.unet.config.in_channels, self.pipe.unet.config.sample_size, self.pipe.unet.config.sample_size),
            generator=gen,
            device=device,
            dtype=pipe_dtype,
        ) * self.pipe.scheduler.init_noise_sigma

        # Text embeddings
        with torch.no_grad():
            full_text_embeddings, uncond_embeddings = _encode_text(self.pipe, prompt, negative_prompt, pipe_dtype)

        # Timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)

        # CG window from tail fraction
        mid_fraction = float(max(0.0, min(1.0, mid_fraction)))
        start_step_idx = int(num_inference_steps * (1.0 - mid_fraction))
        end_step_idx = num_inference_steps

        # mid-block hook
        mid_block_features = {}

        def _mid_hook(_m, _inp, out):
            x = out
            if isinstance(x, (tuple, list)):
                x = x[0]
            if hasattr(x, "sample"):
                x = x.sample
            mid_block_features["output"] = x

        hook = self.pipe.unet.mid_block.register_forward_hook(_mid_hook)

        for i, t in enumerate(self.pipe.scheduler.timesteps):
            # UNet forward (no grad)
            ctx = torch.autocast(device_type=device.type, dtype=pipe_dtype) if (fp16 and device.type == "cuda") else torch.no_grad()
            with ctx:
                latent_in = torch.cat([latents, latents], dim=0)
                latent_in = self.pipe.scheduler.scale_model_input(latent_in, t).to(pipe_dtype)
                noise_pred = self.pipe.unet(latent_in, t, encoder_hidden_states=full_text_embeddings).sample

            # CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # CG
            if safety_scale > 0 and start_step_idx <= i < end_step_idx:
                with torch.enable_grad():
                    latents_grad = latents.detach().clone().to(torch.float32).requires_grad_(True)
                    latent_grad_in = self.pipe.scheduler.scale_model_input(latents_grad.to(pipe_dtype), t)
                    _ = self.pipe.unet(latent_grad_in, t, encoder_hidden_states=uncond_embeddings)

                    feats = mid_block_features.get("output", None)
                    if feats is None:
                        grad = torch.zeros_like(latents_grad)
                    else:
                        feats = feats.to(device=device, dtype=next(classifier.parameters()).dtype)
                        if feats.dim() == 3:
                            feats = feats.unsqueeze(0)

                        logits = classifier(feats)
                        probs = torch.softmax(logits, dim=-1).squeeze(0)

                        if torch.isnan(probs).any() or torch.isinf(probs).any():
                            grad = torch.zeros_like(latents_grad)
                        else:
                            loss = -torch.log(probs[safe_idx].clamp_min(1e-6))
                            grad = torch.autograd.grad(loss, latents_grad, retain_graph=False)[0]

                    latents = latents.detach() - safety_scale * grad.to(latents.dtype)
                    mid_block_features.clear()
                    del latents_grad, feats, logits, probs, grad

            # Step
            latents = self.pipe.scheduler.step(noise_pred_cfg, t, latents).prev_sample

        hook.remove()

        # Decode
        with torch.no_grad():
            latents = latents / self.pipe.vae.config.scaling_factor
            img = self.pipe.vae.decode(latents.to(self.pipe.vae.dtype)).sample
            img = (img / 2 + 0.5).clamp(0, 1)
            img = img[0].permute(1, 2, 0).cpu().float().numpy()

        return Image.fromarray((img * 255).round().astype("uint8"))
