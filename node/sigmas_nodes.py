import math
import torch
import torch.nn.functional as F


class SigmasCantorFlow:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 1, "max": 1000, "step": 1}),
                "start_value": ("FLOAT", {"default": 14.614642, "min": 0.0, "max": 100.0, "step": 0.01}),
                "end_value": ("FLOAT", {"default": 0.0291675, "min": 0.0, "max": 100.0, "step": 0.01}),
                "cantor_mode": (["classic", "resonant", "geometric", "staircase"], {"default": "resonant"}),
                "max_iterations": ("INT", {"default": 8, "min": 1, "max": 16, "step": 1}),
                "phi_resonance": ("FLOAT", {"default": 0.29514, "min": 0.0, "max": 1.0, "step": 0.00001}),
                "geometric_base": ("FLOAT", {"default": 3.0, "min": 2.0, "max": 10.0, "step": 0.1}),
                "sd3_shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1,
                                        "tooltip": "SD3-style timestep shift (higher = bias toward clean). Match training value!"}),
                "apply_shift": ("BOOLEAN", {"default": True, "tooltip": "Apply SD3 shift like David training"}),
                "smoothing": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pad_end": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "RES4LYF/schedulers"
    DESCRIPTION = "Cantor flow scheduler aligned with David's training (with SD3 shift and geometric transitions)."

    def sd3_shift(self, t, shift):
        """Apply SD3-style timestep shift (matches David training)"""
        if shift <= 0:
            return t
        return shift * t / (1.0 + (shift - 1.0) * t)

    def cantor_classic(self, t, max_iter):
        """Classic Cantor function"""
        import torch
        result = torch.zeros_like(t)
        power = 1.0

        for level in range(max_iter):
            t_scaled = (t * (3 ** level)) % 3
            mask = (t_scaled >= 1) & (t_scaled < 2)
            result += torch.where(mask, torch.tensor(power / 2, dtype=t.dtype, device=t.device),
                                  torch.zeros_like(t))
            power /= 2

        return result

    def cantor_resonant(self, t, max_iter, phi):
        """Cantor with phi resonance (0.29514 harmonic)"""
        import torch
        import math
        base_cantor = self.cantor_classic(t, max_iter)

        # Phi resonance standing wave
        resonance = torch.sin(2 * math.pi * t / phi)
        resonance = (resonance + 1) / 2

        # Blend
        modulated = base_cantor * (1 - phi) + resonance * phi

        return modulated

    def cantor_geometric(self, t, max_iter, base):
        """Geometric Cantor (pentachoron base=3)"""
        import torch
        result = torch.zeros_like(t)

        for level in range(max_iter):
            t_scaled = (t * (base ** level)) % base
            interval_size = 1.0 / base
            keep_mask = torch.zeros_like(t, dtype=torch.bool)

            for i in range(int(base)):
                if i != int(base // 2):
                    mask = (t_scaled >= i * interval_size) & (t_scaled < (i + 1) * interval_size)
                    keep_mask |= mask

            result += torch.where(keep_mask, torch.tensor(1.0 / (base ** level), dtype=t.dtype, device=t.device),
                                  torch.zeros_like(t))

        result = result / result.max() if result.max() > 0 else result
        return result

    def cantor_staircase(self, t, max_iter):
        """Cantor devil's staircase"""
        import torch
        result = torch.zeros_like(t)
        left = torch.zeros_like(t)
        right = torch.ones_like(t)

        for _ in range(max_iter):
            mid_left = left + (right - left) / 3
            mid_right = left + 2 * (right - left) / 3

            mask_left = t < mid_left
            mask_right = t >= mid_right
            mask_middle = ~mask_left & ~mask_right

            result = torch.where(mask_middle, (left + right) / 2, result)
            right = torch.where(mask_left, mid_left, right)
            left = torch.where(mask_right, mid_right, left)
            result = torch.where(mask_left | mask_right, (left + right) / 2, result)

        return result

    def apply_smoothing(self, sigmas, smoothing_factor):
        """Apply pentachoron-aware smoothing"""
        import torch
        import torch.nn.functional as F

        if smoothing_factor <= 0:
            return sigmas

        kernel_size = max(3, int(len(sigmas) * smoothing_factor * 0.1))
        if kernel_size % 2 == 0:
            kernel_size += 1

        sigma_gauss = kernel_size / 6.0
        x = torch.arange(kernel_size, dtype=sigmas.dtype, device=sigmas.device)
        x = x - kernel_size // 2
        kernel = torch.exp(-x ** 2 / (2 * sigma_gauss ** 2))
        kernel = kernel / kernel.sum()

        padded = F.pad(sigmas.unsqueeze(0).unsqueeze(0),
                       (kernel_size // 2, kernel_size // 2),
                       mode='reflect')
        smoothed = F.conv1d(padded, kernel.unsqueeze(0).unsqueeze(0))

        return smoothed.squeeze()

    def main(self, steps, start_value, end_value, cantor_mode, max_iterations,
             phi_resonance, geometric_base, sd3_shift, apply_shift, smoothing, pad_end):
        import torch

        # ================================================================
        # STEP 1: Generate base Cantor distribution in [0, 1]
        # ================================================================
        t = torch.linspace(0, 1, steps, dtype=torch.float32)

        if cantor_mode == "classic":
            cantor_values = self.cantor_classic(t, max_iterations)
        elif cantor_mode == "resonant":
            cantor_values = self.cantor_resonant(t, max_iterations, phi_resonance)
        elif cantor_mode == "geometric":
            cantor_values = self.cantor_geometric(t, max_iterations, geometric_base)
        elif cantor_mode == "staircase":
            cantor_values = self.cantor_staircase(t, max_iterations)

        # Normalize to [0, 1]
        if cantor_values.max() > cantor_values.min():
            cantor_values = (cantor_values - cantor_values.min()) / (cantor_values.max() - cantor_values.min())

        # ================================================================
        # STEP 2: Apply SD3 shift (CRITICAL - matches David training!)
        # ================================================================
        if apply_shift:
            # Apply shift element-wise
            cantor_shifted = torch.zeros_like(cantor_values)
            for i in range(len(cantor_values)):
                cantor_shifted[i] = self.sd3_shift(cantor_values[i].item(), sd3_shift)
            cantor_values = cantor_shifted

            # Re-normalize after shift
            if cantor_values.max() > cantor_values.min():
                cantor_values = (cantor_values - cantor_values.min()) / (cantor_values.max() - cantor_values.min())

        # ================================================================
        # STEP 3: Map to sigma range (HIGH to LOW for denoising)
        # ================================================================
        # During denoising, we go from high noise (start_value) to low noise (end_value)
        # Cantor t=0 should map to high sigma (noisy), t=1 should map to low sigma (clean)
        sigmas = start_value * (1 - cantor_values) + end_value * cantor_values

        # ================================================================
        # STEP 4: Apply geometric smoothing (preserve pentachoron structure)
        # ================================================================
        if smoothing > 0:
            sigmas = self.apply_smoothing(sigmas, smoothing)
            # Re-normalize to maintain range
            if sigmas.max() > sigmas.min():
                sigmas = ((sigmas - sigmas.min()) / (sigmas.max() - sigmas.min())) * (
                            start_value - end_value) + end_value

        # ================================================================
        # STEP 5: Ensure monotonic decreasing (critical for samplers!)
        # ================================================================
        # Some Cantor modes may not be perfectly monotonic
        # Force decreasing order for stability
        sigmas = torch.sort(sigmas, descending=True)[0]

        # ================================================================
        # STEP 6: Pad with zero (standard for ComfyUI samplers)
        # ================================================================
        if pad_end:
            sigmas = torch.cat([sigmas, torch.tensor([0.0], dtype=sigmas.dtype)])

        return (sigmas,)

import torch
import torch.nn.functional as F
import math
import comfy.model_sampling
import comfy.samplers
import comfy.sample
from comfy.k_diffusion.sampling import to_d
import latent_preview


class FlowMatchingCantorSampler:
    """
    Flow-matching sampler for David-trained students using Cantor-based scheduling.
    Respects geometric pentachoron transitions and 0.29514 phi resonance.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "sigmas": ("SIGMAS", {"tooltip": "Connect Cantor flow sigmas here"}),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "cfg": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_mode": (["velocity", "flow_matching", "rectified_flow"], {"default": "velocity"}),
                "phi_correction": ("BOOLEAN", {"default": True, "tooltip": "Apply 0.29514 resonance correction"}),
                "geometric_guidance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling/david"

    def sample(self, model, sigmas, latent_image, seed, cfg, sampler_mode,
               phi_correction, geometric_guidance, positive=None, negative=None):

        latent = latent_image["samples"]
        device = latent.device
        dtype = latent.dtype

        # Ensure sigmas are on the correct device
        sigmas = sigmas.to(device=device, dtype=dtype)

        # Get model sampling
        model_sampling = model.get_model_object("model_sampling")

        # Initialize noise on correct device
        torch.manual_seed(seed)
        noise = torch.randn_like(latent, device=device, dtype=dtype)

        # Start from noisy latent
        x = latent + sigmas[0] * noise

        # Sampling loop
        for i in range(len(sigmas) - 1):
            sigma_curr = sigmas[i]
            sigma_next = sigmas[i + 1]

            # Get timestep for model
            t = model_sampling.timestep(sigma_curr).to(device)

            # =============================================================
            # FLOW MATCHING PREDICTION
            # =============================================================
            if sampler_mode == "velocity":
                # Student predicts velocity directly
                v_uncond = self._predict_velocity(
                    model, x, t, negative if negative else None, device, dtype
                )
                v_cond = self._predict_velocity(
                    model, x, t, positive, device, dtype
                )

                # CFG on velocity
                v_pred = v_uncond + cfg * (v_cond - v_uncond)

            elif sampler_mode == "flow_matching":
                # Convert to flow matching parameterization
                eps_uncond = self._predict_noise(
                    model, x, t, negative if negative else None, device, dtype
                )
                eps_cond = self._predict_noise(
                    model, x, t, positive, device, dtype
                )

                eps_pred = eps_uncond + cfg * (eps_cond - eps_uncond)

                # Convert to velocity
                alpha = model_sampling.calculate_input(sigma_curr, noise)
                if not isinstance(alpha, torch.Tensor):
                    alpha = torch.tensor(alpha, device=device, dtype=dtype)
                alpha = alpha.to(device=device, dtype=dtype)

                sigma_scaled = sigma_curr.to(device=device, dtype=dtype)

                # Predict x0
                x0_pred = (x - sigma_scaled * eps_pred) / (alpha + 1e-8)

                # Compute velocity
                v_pred = alpha * eps_pred - sigma_scaled * x0_pred

            elif sampler_mode == "rectified_flow":
                # Rectified flow
                eps_uncond = self._predict_noise(
                    model, x, t, negative if negative else None, device, dtype
                )
                eps_cond = self._predict_noise(
                    model, x, t, positive, device, dtype
                )

                eps_pred = eps_uncond + cfg * (eps_cond - eps_uncond)

                # Estimate x0 and x1
                alpha = model_sampling.calculate_input(sigma_curr, noise)
                if not isinstance(alpha, torch.Tensor):
                    alpha = torch.tensor(alpha, device=device, dtype=dtype)
                alpha = alpha.to(device=device, dtype=dtype)

                x0_pred = (x - sigma_curr * eps_pred) / (alpha + 1e-8)
                x1_pred = noise

                # Rectified flow velocity
                v_pred = x1_pred - x0_pred

            # =============================================================
            # PHI RESONANCE CORRECTION (0.29514)
            # =============================================================
            if phi_correction:
                phi = 0.29514

                # Compute flow coherence
                t_normalized = i / (len(sigmas) - 1)

                # Resonance modulation
                resonance = math.sin(2 * math.pi * t_normalized / phi)
                resonance = (resonance + 1) / 2

                # Apply geometric guidance
                geo_weight = geometric_guidance * resonance

                # Smooth velocity field
                if geo_weight > 0:
                    v_pred = self._geometric_smooth(v_pred, geo_weight)

            # =============================================================
            # INTEGRATION STEP
            # =============================================================
            dt = sigma_next - sigma_curr

            # Adaptive step size based on Cantor curvature
            if i > 0 and i < len(sigmas) - 2:
                d2_sigma = sigmas[i + 1] - 2 * sigmas[i] + sigmas[i - 1]
                curvature = abs(float(d2_sigma))
                step_scale = 1.0 / (1.0 + curvature * 10)
            else:
                step_scale = 1.0

            # Euler step
            x = x + v_pred * dt * step_scale

            # Clamp for stability
            if sigma_next > 0:
                x = x.clamp(-10, 10)

        return ({"samples": x},)

    def _predict_velocity(self, model, x, t, cond, device, dtype):
        """Predict velocity from model."""
        if cond is None:
            cond = [[torch.zeros(1, 77, 768, device=device, dtype=dtype), {}]]

        model_output = model.apply_model(x, t, cond=cond)
        return model_output

    def _predict_noise(self, model, x, t, cond, device, dtype):
        """Predict noise from model."""
        if cond is None:
            cond = [[torch.zeros(1, 77, 768, device=device, dtype=dtype), {}]]

        model_output = model.apply_model(x, t, cond=cond)
        return model_output

    def _geometric_smooth(self, v, strength):
        """Apply pentachoron-aware smoothing."""
        if strength <= 0:
            return v

        device = v.device
        dtype = v.dtype
        kernel_size = 5
        sigma = kernel_size / 6.0

        x = torch.arange(kernel_size, dtype=dtype, device=device)
        x = x - kernel_size // 2
        y = x.view(-1, 1)
        x = x.view(1, -1)

        kernel_2d = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel_2d = kernel_2d / kernel_2d.sum()

        smoothed = torch.zeros_like(v)
        for c in range(v.shape[1]):
            v_c = v[:, c:c + 1]
            v_padded = F.pad(v_c, (2, 2, 2, 2), mode='reflect')
            kernel_4d = kernel_2d.view(1, 1, kernel_size, kernel_size)
            v_smooth = F.conv2d(v_padded, kernel_4d)
            smoothed[:, c:c + 1] = v_smooth

        return (1 - strength) * v + strength * smoothed


class FlowMatchingCantorKSampler:
    """
    KSampler-compatible wrapper for Cantor flow matching.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 100.0}),
                "sampler_name": (["euler", "heun", "dpm_2"],),
                "scheduler": (["cantor_resonant", "cantor_geometric", "cantor_staircase"],),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "latent_image": ("LATENT",),

                # Cantor parameters
                "phi_resonance": ("FLOAT", {"default": 0.29514, "min": 0.0, "max": 1.0, "step": 0.00001}),
                "cantor_iterations": ("INT", {"default": 8, "min": 1, "max": 16}),
                "geometric_base": ("FLOAT", {"default": 3.0, "min": 2.0, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/david"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, denoise,
               latent_image, phi_resonance, cantor_iterations, geometric_base,
               positive=None, negative=None):

        latent = latent_image["samples"]
        device = latent.device
        dtype = latent.dtype

        # Generate Cantor sigmas on the correct device
        sigmas = self._get_cantor_sigmas(
            model, steps, scheduler, phi_resonance,
            cantor_iterations, geometric_base, device, dtype
        )

        # Apply denoise
        if denoise < 1.0:
            steps_actual = int(steps * denoise)
            sigmas = sigmas[-(steps_actual + 1):]

        # Setup noise on correct device
        torch.manual_seed(seed)
        noise = torch.randn_like(latent, device=device, dtype=dtype)
        latent_noised = latent + noise * sigmas[0]

        # Get sampler from comfy
        sampler = comfy.samplers.sampler_object(sampler_name)

        # Create callback for progress
        callback = latent_preview.prepare_callback(model, steps)

        # Sample using ComfyUI's API
        samples = comfy.sample.sample_custom(
            model=model,
            noise=noise,
            cfg=cfg,
            sampler=sampler,
            sigmas=sigmas,
            positive=positive,
            negative=negative,
            latent_image=latent_noised,
            noise_mask=None,
            callback=callback,
            disable_pbar=False,
            seed=seed
        )

        out = latent_image.copy()
        out["samples"] = samples

        return (out,)

    def _get_cantor_sigmas(self, model, steps, mode, phi, iterations, base, device, dtype):
        """Generate Cantor-based sigmas on the correct device."""
        model_sampling = model.get_model_object("model_sampling")
        sigma_min = float(model_sampling.sigma_min)
        sigma_max = float(model_sampling.sigma_max)

        # Create tensor on correct device from the start
        t = torch.linspace(0, 1, steps, dtype=dtype, device=device)

        if mode == "cantor_resonant":
            cantor_values = self._cantor_resonant(t, iterations, phi)
        elif mode == "cantor_geometric":
            cantor_values = self._cantor_geometric(t, iterations, base)
        elif mode == "cantor_staircase":
            cantor_values = self._cantor_staircase(t, iterations)
        else:
            cantor_values = self._cantor_classic(t, iterations)

        if cantor_values.max() > cantor_values.min():
            cantor_values = (cantor_values - cantor_values.min()) / \
                            (cantor_values.max() - cantor_values.min())

        sigmas = sigma_max + (sigma_min - sigma_max) * cantor_values
        sigmas = torch.cat([sigmas, torch.zeros(1, device=device, dtype=dtype)])

        return sigmas

    def _cantor_classic(self, t, max_iter):
        result = torch.zeros_like(t)
        power = 1.0
        for level in range(max_iter):
            t_scaled = (t * (3 ** level)) % 3
            mask = (t_scaled >= 1) & (t_scaled < 2)
            result += torch.where(mask, torch.tensor(power / 2, device=t.device, dtype=t.dtype), torch.zeros_like(t))
            power /= 2
        return result

    def _cantor_resonant(self, t, max_iter, phi):
        base_cantor = self._cantor_classic(t, max_iter)
        resonance = torch.sin(2 * math.pi * t / phi)
        resonance = (resonance + 1) / 2
        return base_cantor * (1 - phi) + resonance * phi

    def _cantor_geometric(self, t, max_iter, base):
        result = torch.zeros_like(t)
        for level in range(max_iter):
            t_scaled = (t * (base ** level)) % base
            interval_size = 1.0 / base
            keep_mask = torch.zeros_like(t, dtype=torch.bool)
            for i in range(int(base)):
                if i != int(base // 2):
                    mask = (t_scaled >= i * interval_size) & \
                           (t_scaled < (i + 1) * interval_size)
                    keep_mask |= mask
            result += torch.where(keep_mask, torch.tensor(1.0 / (base ** level), device=t.device, dtype=t.dtype),
                                  torch.zeros_like(t))
        return result / result.max() if result.max() > 0 else result

    def _cantor_staircase(self, t, max_iter):
        result = torch.zeros_like(t)
        left = torch.zeros_like(t)
        right = torch.ones_like(t)

        for _ in range(max_iter):
            mid_left = left + (right - left) / 3
            mid_right = left + 2 * (right - left) / 3

            mask_left = t < mid_left
            mask_right = t >= mid_right
            mask_middle = ~mask_left & ~mask_right

            result = torch.where(mask_middle, (left + right) / 2, result)
            right = torch.where(mask_left, mid_left, right)
            left = torch.where(mask_right, mid_right, left)
            result = torch.where(mask_left | mask_right, (left + right) / 2, result)

        return result

