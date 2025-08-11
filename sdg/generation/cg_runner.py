# sdg/generation/cg_runner.py
import torch
from PIL import Image
from contextlib import nullcontext

CLASSIFIER_CLASS_NAMES = ['gore', 'hate', 'medical', 'safe', 'sexual']

@torch.no_grad()
def _encode_text(pipe, prompt, negative_prompt, device, dtype):
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

    text_emb   = te(text_ids)[0].to(device=device, dtype=dtype)
    uncond_emb = te(uncond_ids)[0].to(device=device, dtype=dtype)
    return torch.cat([uncond_emb, text_emb], dim=0), uncond_emb


class SafeGuidanceRunner:
    """
    CG-only runner that is safe with CPU/GPU offload.
    """

    def __init__(self, pipe, classifier):
        self.pipe = pipe
        self.classifier = classifier

    def _exec_device(self):
        # With enable_model_cpu_offload(), diffusers sets _execution_device
        dev = getattr(self.pipe, "_execution_device", None)
        if dev is None:
            dev = next(self.pipe.unet.parameters()).device
        return dev

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
        exec_device = self._exec_device()  # the device UNet actually runs on
        pipe_dtype  = self.pipe.unet.dtype

        # Classifier on execution device
        classifier = self.classifier.to(exec_device).eval()

        # Generator & latents on exec_device
        gen = torch.Generator(exec_device).manual_seed(int(seed))
        latents = torch.randn(
            (1, self.pipe.unet.config.in_channels, self.pipe.unet.config.sample_size, self.pipe.unet.config.sample_size),
            generator=gen,
            device=exec_device,
            dtype=pipe_dtype,
        ) * self.pipe.scheduler.init_noise_sigma

        # Timesteps on exec_device
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=exec_device)

        # Text embeddings on exec_device
        with torch.no_grad():
            full_text_embeddings, uncond_embeddings = _encode_text(
                self.pipe, prompt, negative_prompt, exec_device, pipe_dtype
            )

        # CG window (tail fraction)
        mid_fraction = float(max(0.0, min(1.0, mid_fraction)))
        start_step_idx = int(num_inference_steps * (1.0 - mid_fraction))
        end_step_idx = num_inference_steps

        # mid-block hook (features will be on exec_device since UNet runs there)
        mid_block_features = {}
        def _mid_hook(_m, _inp, out):
            x = out[0] if isinstance(out, (tuple, list)) else out
            if hasattr(x, "sample"):
                x = x.sample
            mid_block_features["output"] = x
        hook = self.pipe.unet.mid_block.register_forward_hook(_mid_hook)

        autocast_ctx = (
            torch.autocast(device_type=exec_device.type, dtype=pipe_dtype)
            if (fp16 and exec_device.type == "cuda")
            else nullcontext()
        )

        for i, t in enumerate(self.pipe.scheduler.timesteps):
            # Ensure timestep tensor is on exec_device (it should be, but be explicit)
            t = t.to(exec_device)

            # UNet forward without grad
            with autocast_ctx:
                latent_in = torch.cat([latents, latents], dim=0)  # [2, C, H, W]
                latent_in = self.pipe.scheduler.scale_model_input(latent_in, t)
                latent_in = latent_in.to(device=exec_device, dtype=pipe_dtype)

                noise_out = self.pipe.unet(
                    latent_in, t, encoder_hidden_states=full_text_embeddings
                ).sample

            # CFG on exec_device
            noise_pred_uncond, noise_pred_text = noise_out.chunk(2)
            noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            noise_pred_cfg = noise_pred_cfg.to(device=exec_device, dtype=latents.dtype)

            # CG
            if safety_scale > 0 and start_step_idx <= i < end_step_idx:
                with torch.enable_grad():
                    latents_grad = latents.detach().clone().to(exec_device, dtype=torch.float32).requires_grad_(True)
                    latent_grad_in = self.pipe.scheduler.scale_model_input(latents_grad.to(pipe_dtype), t)
                    _ = self.pipe.unet(latent_grad_in, t, encoder_hidden_states=uncond_embeddings)

                    feats = mid_block_features.get("output", None)
                    if feats is None:
                        grad = torch.zeros_like(latents_grad)
                    else:
                        feats = feats.to(device=exec_device, dtype=next(classifier.parameters()).dtype)
                        if feats.dim() == 3:
                            feats = feats.unsqueeze(0)
                        logits = classifier(feats)
                        probs  = torch.softmax(logits, dim=-1).squeeze(0)
                        if torch.isnan(probs).any() or torch.isinf(probs).any():
                            grad = torch.zeros_like(latents_grad)
                        else:
                            loss = -torch.log(probs[safe_idx].clamp_min(1e-6))
                            grad = torch.autograd.grad(loss, latents_grad, retain_graph=False)[0]

                    latents = latents.detach() - safety_scale * grad.to(latents.dtype, device=exec_device)
                    mid_block_features.clear()
                    del latents_grad, feats, logits, probs, grad

            # Scheduler step — all on exec_device
            latents = self.pipe.scheduler.step(noise_pred_cfg, t, latents).prev_sample

        hook.remove()

        # Decode on VAE device
        vae_device = next(self.pipe.vae.parameters()).device
        with torch.no_grad():
            latents = latents / self.pipe.vae.config.scaling_factor
            img = self.pipe.vae.decode(latents.to(vae_device, dtype=self.pipe.vae.dtype)).sample
            img = (img / 2 + 0.5).clamp(0, 1)
            img = img[0].permute(1, 2, 0).cpu().float().numpy()

        return Image.fromarray((img * 255).round().astype("uint8"))
