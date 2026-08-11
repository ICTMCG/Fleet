"""DINOv3 + Xception dual-branch model: high-frequency / semantic branches fused via multi-head routing, with LoRA fine-tuning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fleet.models.xception import Xception


class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) layer."""

    def __init__(self, original_layer, rank=8, alpha=16, dropout=0.0):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        for param in original_layer.parameters():
            param.requires_grad = False

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return original_output + lora_output


def apply_lora_to_model(model, rank=8, alpha=16, dropout=0.0, target_modules=None, exclude_modules=None):
    """Apply LoRA to the model's MLP layers (excluding q/k/v/out matrices)."""
    if exclude_modules is None:
        exclude_modules = ["query", "key", "value", "output", "q_proj", "k_proj", "v_proj", "o_proj"]

    lora_modules = []

    def apply_lora_recursive(module, name_prefix=""):
        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name

            should_exclude = any(exclude_name in full_name.lower() for exclude_name in exclude_modules)
            is_mlp_layer = "mlp" in full_name.lower() or "feedforward" in full_name.lower()

            if isinstance(child, nn.Linear) and not should_exclude and is_mlp_layer:
                lora_layer = LoRALayer(child, rank=rank, alpha=alpha, dropout=dropout)
                setattr(module, name, lora_layer)
                lora_modules.append((full_name, lora_layer))
            else:
                apply_lora_recursive(child, full_name)

    apply_lora_recursive(model)
    return model, lora_modules


class MultiHeadRouting(nn.Module):
    """Multi-head routing fusion: the Xception features produce routing signals that gate the per-head DINOv3 features."""

    def __init__(self, embed_dim=1024, num_heads=8, dropout=0.1, q_dim=128):
        super(MultiHeadRouting, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.q_dim = q_dim
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert self.head_dim == q_dim, "head_dim must equal q_dim"

        self.q_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.LayerNorm(embed_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, q_dim),
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.q_dim**-0.5

    def forward(self, signal, feature):
        batch_size = signal.size(0)

        Q = self.q_proj(signal)
        K = self.k_proj(feature)
        V = self.v_proj(feature)

        K_segments = K.view(batch_size, self.num_heads, self.head_dim)
        V_segments = V.view(batch_size, self.num_heads, self.head_dim)

        scores = torch.bmm(Q.unsqueeze(1), K_segments.transpose(1, 2)).squeeze(1) * self.scale
        routing_weights = self.dropout(F.softmax(scores, dim=-1))

        routed = (routing_weights.unsqueeze(-1) * V_segments).view(batch_size, self.embed_dim)
        output = self.out_proj(routed)
        return output, routing_weights


class DualBranchContrastiveModel(nn.Module):
    """Dual-branch contrastive model: DINOv3 + Xception + multi-head routing fusion + projection head."""

    def __init__(
        self,
        dinov3_model_path,
        xception_feature_dim=1024,
        projection_dim=512,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        num_heads=8,
        q_dim=128,
        feature_layer=20,
        use_last_hidden_state=False,
        normalize_feature=False,
    ):
        super(DualBranchContrastiveModel, self).__init__()

        # DINOv3 feature extraction:
        #   - use_last_hidden_state=True  -> outputs.last_hidden_state[:,0,:] (i.e. m.norm(hidden_states[-1]), normalized)
        #   - use_last_hidden_state=False, normalize_feature=False -> hidden_states[feature_layer][:,0,:] (raw layer output, not normalized)
        #   - use_last_hidden_state=False, normalize_feature=True  -> m.norm(hidden_states[feature_layer])[:,0,:] (apply the final LayerNorm to the selected layer)
        #   Note: for DINOv3, hidden_states[-1] != last_hidden_state (the former is not passed through the final LayerNorm m.norm).
        self.feature_layer = feature_layer
        self.use_last_hidden_state = use_last_hidden_state
        self.normalize_feature = normalize_feature

        from transformers import AutoModel

        self.dinov3 = AutoModel.from_pretrained(dinov3_model_path, trust_remote_code=True)
        for param in self.dinov3.parameters():
            param.requires_grad = False

        self.dinov3, self.lora_modules = apply_lora_to_model(
            self.dinov3,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )

        dinov3_dim = self.dinov3.config.hidden_size
        self.dinov3_proj = nn.Linear(dinov3_dim, 1024) if dinov3_dim != 1024 else nn.Identity()

        self.xception = Xception(in_channels=3, num_classes=2)
        self.xception_fc = nn.Linear(2048, 1024)

        self.routing = MultiHeadRouting(
            embed_dim=1024,
            num_heads=num_heads,
            dropout=0.1,
            q_dim=q_dim,
        )

        self.projection = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, projection_dim),
        )

    def forward(self, x_dinov3, x_xception):
        if x_xception.dim() == 5:
            x_xception = x_xception.squeeze(1)
        assert x_xception.dim() == 4, f"x_xception should be a 4D tensor, got {x_xception.shape}"

        # DINOv3 features: optional last_hidden_state (normalized) or hidden_states[feature_layer] (raw)
        if self.use_last_hidden_state:
            outputs = self.dinov3(x_dinov3)
            dinov3_features = outputs.last_hidden_state[:, 0, :]
        else:
            # hidden_states[0] is the embedding output, so layer i is hidden_states[i]
            outputs = self.dinov3(x_dinov3, output_hidden_states=True)
            feat = outputs.hidden_states[self.feature_layer]  # [batch, seq, dim]
            if self.normalize_feature:
                # Apply DINOv3's final LayerNorm (m.norm) to the selected layer
                feat = self.dinov3.norm(feat)
            dinov3_features = feat[:, 0, :]
        dinov3_features = self.dinov3_proj(dinov3_features)

        xception_features = self.xception.features(x_xception)
        xception_features = F.adaptive_avg_pool2d(xception_features, (1, 1))
        xception_features = xception_features.view(xception_features.size(0), -1)
        xception_features = self.xception_fc(xception_features)

        fused_features, routing_weights = self.routing(xception_features, dinov3_features)

        projected = self.projection(fused_features)
        projected = F.normalize(projected, p=2, dim=1)

        return projected, routing_weights
