"""Utility functions: paths, environment variables, image transforms, losses, dynamic imports, plotting helpers."""

from __future__ import annotations

import importlib
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# --------------------------------------------------------------------------- #
# Paths (DINOv3, etc.)
# --------------------------------------------------------------------------- #

DINOV3_ENV_KEY = "FLEET_DINOV3_MODEL_PATH"
DEFAULT_DINOV3_SUBDIR = Path("weights") / "dinov3-vitl16-pretrain-lvd1689m"


def repo_root() -> Path:
    """``src/fleet/utils.py`` -> repository root."""
    return Path(__file__).resolve().parents[2]


def default_dinov3_model_dir() -> Path:
    return repo_root() / DEFAULT_DINOV3_SUBDIR


def resolve_dinov3_model_path(explicit: str | None = None) -> str:
    """
    Resolve the DINOv3 weights/config directory.

    Priority: explicit > ``FLEET_DINOV3_MODEL_PATH`` > ``<repo>/weights/dinov3-vitl16-pretrain-lvd1689m``.
    """
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    env = (os.environ.get(DINOV3_ENV_KEY) or "").strip()
    if env:
        return env
    return str(default_dinov3_model_dir().resolve())


# --------------------------------------------------------------------------- #
# Environment variables / CLI
# --------------------------------------------------------------------------- #


def env_path(key: str) -> str:
    return (os.environ.get(key) or "").strip()


def require_env(key: str) -> str:
    v = env_path(key)
    if not v:
        raise SystemExit(f"Missing config: please set the environment variable {key} (see the repo .env.example).")
    return v


def coalesce_cli_env(arg_val, env_key: str) -> str:
    return (arg_val or "").strip() or (os.environ.get(env_key) or "").strip()


# --------------------------------------------------------------------------- #
# Image / tensor transforms
# --------------------------------------------------------------------------- #


def apply_fft_high_freq_mask(img_tensor, xception_crop_size=128):
    """Apply an FFT high-frequency mask to a [3,H,W] tensor (in [0,1]) and return the ImageNet-normalized Xception input."""
    if img_tensor.shape[-1] != xception_crop_size or img_tensor.shape[-2] != xception_crop_size:
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0),
            size=(xception_crop_size, xception_crop_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    img_fft = torch.fft.fft2(img_tensor, dim=(-2, -1))
    img_fft_shifted = torch.fft.fftshift(img_fft, dim=(-2, -1))

    h, w = img_fft_shifted.shape[-2:]
    center_h, center_w = h // 2, w // 2
    total_area = h * w
    mask_radius = np.sqrt(total_area * 0.5 / np.pi)

    y, x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=img_tensor.device),
        torch.arange(w, dtype=torch.float32, device=img_tensor.device),
        indexing="ij",
    )
    dist_from_center = torch.sqrt((x - center_w) ** 2 + (y - center_h) ** 2)
    high_freq_mask = (dist_from_center > mask_radius).float().unsqueeze(0)

    img_fft_masked = img_fft_shifted * high_freq_mask
    img_fft_ishifted = torch.fft.ifftshift(img_fft_masked, dim=(-2, -1))
    img_reconstructed = torch.real(torch.fft.ifft2(img_fft_ishifted, dim=(-2, -1)))
    img_reconstructed = torch.clamp(img_reconstructed, 0, 1)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalize(img_reconstructed)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #


def pretrain_contrastive_loss(features, labels, temperature=0.07):
    batch_size = features.size(0)
    similarity_matrix = torch.matmul(features, features.T) / temperature
    labels = labels.unsqueeze(1)
    positive_mask = (labels == labels.T).float()
    positive_mask.fill_diagonal_(0)
    num_positives = torch.clamp(positive_mask.sum(dim=1, keepdim=True), min=1)
    positive_similarities = (similarity_matrix * positive_mask).sum(dim=1, keepdim=True) / num_positives
    exp_similarities = torch.exp(similarity_matrix)
    diag_mask = torch.eye(batch_size, device=features.device, dtype=torch.bool)
    exp_similarities_no_diag = exp_similarities.clone()
    exp_similarities_no_diag[diag_mask] = 0
    sum_exp_similarities = exp_similarities_no_diag.sum(dim=1, keepdim=True)
    return -positive_similarities.mean() + torch.log(sum_exp_similarities + 1e-8).mean()


def attention_orthogonal_loss(attn_weights, labels):
    fake_mask = labels == 0
    real_mask = labels == 1
    if fake_mask.sum().item() == 0 or real_mask.sum().item() == 0:
        return torch.tensor(0.0, device=attn_weights.device, requires_grad=True)
    attn_fake = attn_weights[fake_mask]
    attn_real = attn_weights[real_mask]
    return torch.sum(attn_fake.mean(dim=0) * attn_real.mean(dim=0))


def attention_coverage_loss(attn_weights, labels, epsilon=1e-6):
    fake_mask = labels == 0
    real_mask = labels == 1
    if fake_mask.sum().item() == 0 or real_mask.sum().item() == 0:
        return torch.tensor(0.0, device=attn_weights.device, requires_grad=True)
    W_fake_mean = attn_weights[fake_mask].mean(dim=0)
    W_real_mean = attn_weights[real_mask].mean(dim=0)
    return -torch.mean(torch.log(W_real_mean + W_fake_mean + epsilon))


def fewshot_contrastive_loss(features, labels, temperature=0.07):
    batch_size = features.shape[0]
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float()
    similarity_matrix = torch.matmul(features, features.T) / temperature
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(mask.device),
        0,
    )
    mask = mask * logits_mask
    exp_logits = torch.exp(similarity_matrix) * logits_mask
    log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    return -mean_log_prob_pos.mean()


def distillation_loss(current_features, original_features, weight=1.0):
    cosine_sim = torch.sum(current_features * original_features, dim=1)
    return weight * torch.mean(1.0 - cosine_sim)


def mutual_avoidance_loss(attn_weights, labels, fake_anchor, real_anchor, weight=1.0):
    fake_mask = labels == 0
    real_mask = labels == 1
    loss = torch.tensor(0.0, device=attn_weights.device)
    if fake_mask.sum() > 0:
        fake_attn = attn_weights[fake_mask]
        loss = loss + torch.mean(torch.sum(fake_attn * real_anchor.unsqueeze(0), dim=1))
    if real_mask.sum() > 0:
        real_attn = attn_weights[real_mask]
        loss = loss + torch.mean(torch.sum(real_attn * fake_anchor.unsqueeze(0), dim=1))
    return weight * loss


# --------------------------------------------------------------------------- #
# q-dim inference and dynamic import
# --------------------------------------------------------------------------- #


def infer_q_dim_and_spec_from_paths(paths):
    for p in paths:
        if not p:
            continue
        m = re.search(r"_q(\d+(?:_[A-Za-z0-9]+)*)", str(p))
        if not m:
            continue
        q_spec = m.group(1)
        q_dim_str = q_spec.split("_", 1)[0]
        try:
            q_dim = int(q_dim_str)
        except ValueError:
            continue
        return q_dim, q_spec
    return None, None


def load_q_train_module(q_spec: str, fallback_q_dim: int | None = None):
    tried = []

    def _try_import(spec: str):
        module_name = f"fleet.train.dual_branch_q{spec}"
        tried.append(module_name)
        return importlib.import_module(module_name)

    try:
        return _try_import(str(q_spec))
    except Exception as e_primary:
        if fallback_q_dim is not None and str(fallback_q_dim) != str(q_spec):
            try:
                return _try_import(str(fallback_q_dim))
            except Exception as e_fallback:
                raise RuntimeError(
                    "Failed to import the training script module.\n"
                    f"- First tried: {tried[0]}\n"
                    f"- Then tried: {tried[1]}\n"
                    "Please confirm you have run `pip install -e .` and that the corresponding `dual_branch_q*.py` exists under `src/fleet/train/`.\n"
                    f"Original error (first): {e_primary}\n"
                    f"Original error (fallback): {e_fallback}"
                )
        raise RuntimeError(
            "Failed to import the training script module.\n"
            f"- Tried: {tried[0]}\n"
            "Please confirm you have run `pip install -e .` and that the corresponding `dual_branch_q*.py` exists under `src/fleet/train/`.\n"
            f"Original error: {e_primary}"
        )


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #


def sanitize_dataset_name_for_plot(dataset_name: str) -> str:
    name = dataset_name.replace("Query集", "Query Set")
    name = name.replace("验证集", "Validation Set")
    name = name.replace("训练集", "Training Set")
    # Keep backward-compatible replacements above; non-ASCII chars become "_".
    return "".join(char if ord(char) < 128 else "_" for char in name)
