import math

import torch

import comfy
import logging

from comfy.ldm.modules.diffusionmodules.util import make_beta_schedule
from comfy_extras.nodes_model_advanced import LCM, ModelSamplingDiscreteDistilled

logger = logging.getLogger(__name__)

class AModelSamplingDiscrete:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "sampling": (["eps", "v_prediction", "lcm", "x0", "img_to_img"],),
                              "zsnr": ("BOOLEAN", {"default": False}),
                              }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model, sampling, zsnr):
        m = model.clone()

        sampling_base = comfy.model_sampling.ModelSamplingDiscrete
        if sampling == "eps":
            sampling_type = EPS
        elif sampling == "v_prediction":
            sampling_type = V_PREDICTION
        elif sampling == "lcm":
            sampling_type = LCM
            sampling_base = ModelSamplingDiscreteDistilled
        elif sampling == "x0":
            sampling_type = X0
        elif sampling == "img_to_img":
            sampling_type = IMG_TO_IMG

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config, zsnr=zsnr)

        m.add_object_patch("model_sampling", model_sampling)
        return (m, )


def rescale_zero_terminal_snr_sigmas(
    sigmas: torch.Tensor,
    *,
    min_delta: float = 1.0e-5,
    eps_alpha_bar: float = 4.8973451890853435e-8,
) -> torch.Tensor:
    """
    Rescale `sigmas` so that the cumulative product of alphas (ᾱ) is
    shifted to zero at the last step and re-scaled to keep the first step
    unchanged – *with* fallback protection.

    Parameters
    ----------
    sigmas : 1-D `torch.Tensor`
        σ schedule (noise-to-signal ratio) in ascending time order.
    min_delta : float, optional
        Minimum allowed difference between √ᾱ₀ and √ᾱ_T.  If the actual
        difference is smaller, we 'impulse' the tail by subtracting
        `min_delta` before scaling to avoid a near-zero denominator.
    eps_alpha_bar : float, optional
        Floor value for ᾱ at the terminal step (keeps σ finite in fp32).

    Returns
    -------
    torch.Tensor
        A new σ schedule after zero-terminal-SNR rescaling.
    """
    logger.info("Rescaling sigmas with zero-terminal-SNR impulse: min_delta=%f, eps_alpha_bar=%e", min_delta, eps_alpha_bar)
    debug = {}
    # ᾱ_t = 1 / (1+σ²)
    alphas_bar = 1.0 / (sigmas * sigmas + 1.0)
    alphas_bar_sqrt = torch.sqrt(alphas_bar)

    a0 = alphas_bar_sqrt[0].clone()   # √ᾱ₀
    aT = alphas_bar_sqrt[-1].clone()  # √ᾱ_T  (pre-shift)

    delta = a0 - aT
    if delta.abs() < min_delta:
        # Too small → would explode scaling factor; impulse the tail.
        impulse = min_delta if delta >= 0 else -min_delta
        aT = aT - impulse
        delta = a0 - aT
        debug['impulse'] = { 'min_delta': min_delta, 'impulse': impulse }

    # Shift so last step becomes zero
    alphas_bar_sqrt = alphas_bar_sqrt - aT
    # Scale so first step returns to original value
    alphas_bar_sqrt = alphas_bar_sqrt * (a0 / delta)

    # Re-square to get ᾱ, then replace final ᾱ with floor value
    alphas_bar = alphas_bar_sqrt.pow(2)
    alphas_bar[-1] = eps_alpha_bar    # keep numeric stability

    # Convert back to σ
    sigmas_out = torch.sqrt((1.0 - alphas_bar) / alphas_bar)
    if debug.get('impulse', None) is not None:
        logger.info(f"Rescaled sigmas with zero-terminal-SNR impulse: {debug['impulse']['impulse']}" + f" {debug['min_delta']}")

    return sigmas_out



class ModelSamplingDiscrete(torch.nn.Module):
    def __init__(self, model_config=None, zsnr=None):
        super().__init__()

        if model_config is not None:
            sampling_settings = model_config.sampling_settings
        else:
            sampling_settings = {}

        beta_schedule = sampling_settings.get("beta_schedule", "linear")
        linear_start = sampling_settings.get("linear_start", 0.00085)
        linear_end = sampling_settings.get("linear_end", 0.012)
        timesteps = sampling_settings.get("timesteps", 1000)

        if zsnr is None:
            zsnr = sampling_settings.get("zsnr", False)

        self._register_schedule(given_betas=None, beta_schedule=beta_schedule, timesteps=timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=8e-3, zsnr=zsnr)
        self.sigma_data = 1.0

    def _register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3, zsnr=False):
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        self.zsnr = zsnr

        # self.register_buffer('betas', torch.tensor(betas, dtype=torch.float32))
        # self.register_buffer('alphas_cumprod', torch.tensor(alphas_cumprod, dtype=torch.float32))
        # self.register_buffer('alphas_cumprod_prev', torch.tensor(alphas_cumprod_prev, dtype=torch.float32))

        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        if self.zsnr:
            sigmas = rescale_zero_terminal_snr_sigmas(sigmas)

        self.set_sigmas(sigmas)

    def set_sigmas(self, sigmas):
        self.register_buffer('sigmas', sigmas.float())
        self.register_buffer('log_sigmas', sigmas.log().float())

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)

    def sigma(self, timestep):
        t = torch.clamp(timestep.float().to(self.log_sigmas.device), min=0, max=(len(self.sigmas) - 1))
        low_idx = t.floor().long()
        high_idx = t.ceil().long()
        w = t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp().to(timestep.device)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 999999999.9
        if percent >= 1.0:
            return 0.0
        percent = 1.0 - percent
        return self.sigma(torch.tensor(percent * 999.0)).item()


class EPS:
    def calculate_input(self, sigma, noise):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        return noise / (sigma ** 2 + self.sigma_data ** 2) ** 0.5

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        if max_denoise:
            noise = noise * torch.sqrt(1.0 + sigma ** 2.0)
        else:
            noise = noise * sigma

        noise += latent_image
        return noise

    def inverse_noise_scaling(self, sigma, latent):
        return latent

class V_PREDICTION(EPS):
    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input * self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2) - model_output * sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5

class EDM(V_PREDICTION):
    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input * self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2) + model_output * sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5

class CONST:
    def calculate_input(self, sigma, noise):
        return noise

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        return sigma * noise + (1.0 - sigma) * latent_image

    def inverse_noise_scaling(self, sigma, latent):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (latent.ndim - 1))
        return latent / (1.0 - sigma)

class X0(EPS):
    def calculate_denoised(self, sigma, model_output, model_input):
        return model_output

class IMG_TO_IMG(X0):
    def calculate_input(self, sigma, noise):
        return noise