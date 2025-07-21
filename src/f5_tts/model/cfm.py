"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)
from contextlib import contextmanager

import numpy as np

class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()
        # raw wave

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # text

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # duration

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, cond, torch.zeros_like(cond)
        )  # allow direct control (cut cond audio) with lens passed in

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow (cond)
            if cfg_strength < 1e-5:
                pred = self.transformer(
                    x=x,
                    cond=step_cond,
                    text=text,
                    time=t,
                    mask=mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    cache=True,
                )
                return pred

            # predict flow (cond and uncond), for classifier-free guidance
            pred_cfg = self.transformer(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                cfg_infer=True,
                cache=True,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg_strength

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        if t_start == 0 and use_epss:  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(steps, device=self.device, dtype=step_cond.dtype)
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        *,
        lens: int["b"] | None = None,  # noqa: F821
        noise_scheduler: str | None = None,
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # lens and mask
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)

        mask = lens_to_mask(lens, length=seq_len)  # useless here, as collate_fn will pad to max length in batch

        # get a random span to mask out for training conditionally
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # apply mask will use more memory; might adjust batchsize or batchsampler long sequence threshold
        pred = self.transformer(
            x=φ, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text, mask=mask
        )

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]

        return loss.mean(), cond, pred


class MeanFlowTTS(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(method="euler"),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
        # MeanFlow specific parameters
        flow_ratio=0.5,  # ratio of samples where r != t
        time_dist=['lognorm', -0.4, 1.0],  # time sampling distribution
        adaptive_loss_gamma=0.5,  # for adaptive L2 loss
        jvp_api='autograd',  # 'autograd' or 'functorch'
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer - must support (x, t, r, cond, text) inputs
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

        # MeanFlow specific
        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.adaptive_loss_gamma = adaptive_loss_gamma
        
        # JVP function selection
        assert jvp_api in ['functorch', 'autograd']
        if jvp_api == 'functorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        else:
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True

    @property
    def device(self):
        return next(self.parameters()).device

    def sample_t_r(self, batch_size, device):
        """Sample time pairs (t, r) for MeanFlow"""
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)
        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[1], self.time_dist[2]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))  # sigmoid

        # Ensure r <= t
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        # Set r = t for a portion of samples
        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device, dtype=torch.float32)
        r = torch.tensor(r_np, device=device, dtype=torch.float32)
        return t, r

    def adaptive_l2_loss(self, error):
        """Adaptive L2 loss as in MeanFlow paper"""
        # Reshape error to (batch, -1) and compute norm per sample
        batch_size = error.shape[0]
        error_flat = error.view(batch_size, -1)
        delta_sq = torch.mean(error_flat ** 2, dim=1)
        
        p = 1.0 - self.adaptive_loss_gamma
        c = 1e-3
        w = 1.0 / (delta_sq + c).pow(p)
        loss = delta_sq
        return (w.detach() * loss).mean()

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],
        text: int["b nt"] | list[str],
        duration: int | int["b"],
        *,
        lens: int["b"] | None = None,
        steps=32,
        cfg_strength=1.0,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,
        sway_sampling_coef=None,  # Add for compatibility
    ):
        """MeanFlow sampling - can use multiple steps or single step"""
        self.eval()
        
        # Process conditioning
        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)
        batch, cond_seq_len, device = *cond.shape[:2], cond.device

        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # Process text
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)

        # Handle duration
        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # Pad conditioning
        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        cond_mask = lens_to_mask(lens)
        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

        # Initialize noise
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=device, dtype=cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        # Time steps for sampling
        t_vals = torch.linspace(1.0, 0.0, steps + 1, device=device)

        # MeanFlow sampling
        z = y0
        for i in range(steps):
            t = torch.full((batch,), t_vals[i], device=device)
            r = torch.full((batch,), t_vals[i + 1], device=device)
            
            # Get average velocity
            if cfg_strength < 1e-5:
                u = self.transformer(
                    x=z, t=t, r=r, cond=step_cond, text=text,
                    drop_audio_cond=False, drop_text=False
                )
            else:
                # CFG inference
                u_cond = self.transformer(
                    x=z, t=t, r=r, cond=step_cond, text=text,
                    drop_audio_cond=False, drop_text=False
                )
                u_uncond = self.transformer(
                    x=z, t=t, r=r, cond=step_cond, text=text,
                    drop_audio_cond=True, drop_text=True
                )
                u = u_cond + (u_cond - u_uncond) * cfg_strength
            
            # Update z using average velocity
            dt = (t_vals[i] - t_vals[i + 1]).item()
            z = z - dt * u

        out = torch.where(cond_mask, cond, z)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, None  # trajectory tracking could be added

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],
        text: int["b nt"] | list[str],
        *,
        lens: int["b"] | None = None,
        # CFG parameters
        cfg_scale: float = 2.0,  # guidance scale
        cfg_uncond_mode: str = 'u',  # 'v' or 'u' for unconditional samples,
        noise_scheduler: str | None = None,
    ):
        """MeanFlow training forward pass with proper CFG handling"""
        # Process input
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, self.device

        # Process text
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)

        # Lens and mask
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)

        mask = lens_to_mask(lens, length=seq_len)

        # Get random span mask for conditioning
        frac_lengths = torch.zeros((batch,), device=device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)
        if exists(mask):
            rand_span_mask &= mask

        # Sample time steps
        t, r = self.sample_t_r(batch, device)

        # Prepare data
        x1 = inp  # target mel
        x0 = torch.randn_like(x1)  # noise
        
        # Create interpolation
        t_expanded = t.view(batch, 1, 1)
        z = (1 - t_expanded) * x0 + t_expanded * x1
        
        # Conditional velocity (v = x1 - x0)
        v = x1 - x0

        # Masking for conditioning
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # CFG mask for dropping conditioning
        cfg_drop_mask = torch.rand(batch, device=device) < self.cond_drop_prob
        
        # Prepare CFG velocity
        v_hat = v.clone()
        
        if cfg_scale > 1e-5 and cfg_drop_mask.any():
            # For samples that will be unconditional, we need to modify v_hat
            with torch.no_grad():
                # Create unconditional text (could be padding tokens or a special uncond token)
                uncond_text = torch.zeros_like(text)  # or use a special token
                
                # Get u(z, t, t) for CFG - note we use t for both arguments
                u_t_uncond = self.transformer(
                    x=z,
                    t=t,
                    r=t,  # Important: r = t for the boundary case
                    cond=torch.zeros_like(cond),  # No audio conditioning
                    text=uncond_text,  # No text conditioning
                    drop_audio_cond=True,
                    drop_text=True,
                    mask=mask
                )
                
                # Compute guided velocity
                # v_hat = w * v + (1 - w) * u_t
                w = cfg_scale
                v_hat_guided = w * v + (1 - w) * u_t_uncond
                
                # Apply CFG only to dropped samples
                cfg_drop_mask_expanded = cfg_drop_mask.view(batch, 1, 1)
                
                if cfg_uncond_mode == 'v':
                    # Use original v for unconditional samples (official JAX implementation)
                    v_hat = torch.where(cfg_drop_mask_expanded, v, v_hat_guided)
                else:
                    # Use guided velocity for all CFG samples
                    v_hat = torch.where(cfg_drop_mask_expanded, v_hat_guided, v)

        # Determine which samples to drop conditioning for
        drop_audio_cond_per_sample = torch.rand(batch, device=device) < self.audio_drop_prob
        drop_text_per_sample = cfg_drop_mask  # Text is dropped when doing CFG
        
        # For fully unconditional samples
        full_uncond_mask = drop_audio_cond_per_sample | drop_text_per_sample

        # Prepare conditioning for the main forward pass
        final_cond = cond.clone()
        final_text = text.clone()
        
        # Apply conditioning drops
        for i in range(batch):
            if full_uncond_mask[i]:
                final_cond[i] = torch.zeros_like(cond[i])
                final_text[i] = torch.zeros_like(text[i])  # or special uncond token
            elif drop_audio_cond_per_sample[i]:
                final_cond[i] = torch.zeros_like(cond[i])

        # Define model forward function for JVP
        def model_fn(z_inp, t_inp, r_inp):
            return self.transformer(
                x=z_inp,
                t=t_inp,
                r=r_inp,
                cond=final_cond,
                text=final_text,
                drop_audio_cond=False,  # Already handled above
                drop_text=False,  # Already handled above
                mask=mask
            )

        # Compute JVP with the modified velocity
        # Tangent vectors: (v_hat for z, 1 for t, 0 for r)
        tangents = (v_hat, torch.ones_like(t), torch.zeros_like(r))
        
        if self.create_graph:
            u, dudt = self.jvp_fn(model_fn, (z, t, r), tangents, create_graph=True)
        else:
            u, dudt = self.jvp_fn(model_fn, (z, t, r), tangents)

        # Compute target with v_hat
        t_minus_r = (t - r).view(batch, 1, 1)
        u_target = v_hat - t_minus_r * dudt

        # Compute loss only on masked region
        error = u - u_target.detach()
        error_masked = error[rand_span_mask]

        # Use adaptive L2 loss
        loss = self.adaptive_l2_loss(error_masked)

        return loss, cond, u