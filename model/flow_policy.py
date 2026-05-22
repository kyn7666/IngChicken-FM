# -*- coding: utf-8 -*-
"""
Flow Matching Policy: migrated from DDPM DiffusionPolicy.

Key changes vs. diffusion_policy.py:
  - No DDPM scheduler buffers (betas, alphas, alphas_cumprod, etc.)
  - compute_loss(): FM linear interpolation + velocity regression
  - predict_action(): Euler integration (replaces DDPM/DDIM reverse chain)
  - timestep t is continuous [0,1], scaled to [0,1000] for SinusoidalPosEmb
  - vector_field_net predicts the velocity field (replaces noise_pred_net)
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Vision Encoder  (unchanged from diffusion_policy.py)
# ------------------------------------------------------------------

class SpatialSoftmax(nn.Module):
    def __init__(self, h, w, num_kp):
        super().__init__()
        self.num_kp = num_kp
        self.conv = nn.Conv2d(512, num_kp, 1)
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
        )
        self.register_buffer("pos_x", pos_x.reshape(-1))
        self.register_buffer("pos_y", pos_y.reshape(-1))

    def forward(self, x):
        B, C, H, W = x.shape
        features = self.conv(x)
        features = features.reshape(B, self.num_kp, -1)
        attention = F.softmax(features, dim=-1)
        kp_x = (attention * self.pos_x).sum(dim=-1)
        kp_y = (attention * self.pos_y).sum(dim=-1)
        return torch.cat([kp_x, kp_y], dim=-1)  # (B, num_kp*2)


class VisionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        import torchvision.models as models

        if cfg["type"] == "resnet18":
            backbone = models.resnet18(pretrained=cfg.get("pretrained", True))
        else:
            backbone = models.resnet34(pretrained=cfg.get("pretrained", True))

        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        if cfg.get("spatial_softmax", False):
            num_kp = cfg.get("num_kp", 32)
            self.pool = SpatialSoftmax(4, 4, num_kp)
            self.output_dim = num_kp * 2
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.output_dim = 512

    def forward(self, x):
        """x: (B, T, C, H, W) -> (B, T, D)"""
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        features = self.backbone(x)
        if isinstance(self.pool, SpatialSoftmax):
            out = self.pool(features)
        else:
            out = self.pool(features).flatten(1)
        return out.reshape(B, T, -1)


# ------------------------------------------------------------------
# 1-D Conditional UNet  (unchanged from diffusion_policy.py)
# ------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t is float (scaled to [0,1000] externally for FM)
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConditionalResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=5, n_groups=8,
                 cond_predict_scale=False):
        super().__init__()
        self.cond_predict_scale = cond_predict_scale

        def _compatible_groups(channels, desired):
            g = min(desired, channels)
            while channels % g != 0:
                g -= 1
            return max(g, 1)

        g_out = _compatible_groups(out_channels, n_groups)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.GroupNorm(g_out, out_channels),
                nn.Mish(),
            ),
            nn.Sequential(
                nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.GroupNorm(g_out, out_channels),
                nn.Mish(),
            ),
        ])

        cond_out = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_out))

        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        out = self.blocks[0](x)
        cond_emb = self.cond_encoder(cond)
        if self.cond_predict_scale:
            scale, bias = cond_emb.chunk(2, dim=-1)
            out = out * (scale.unsqueeze(-1) + 1) + bias.unsqueeze(-1)
        else:
            out = out + cond_emb.unsqueeze(-1)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUNet1D(nn.Module):
    def __init__(self, input_dim, cond_dim, down_dims, kernel_size=5, n_groups=8,
                 cond_predict_scale=False):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        diffusion_step_embed_dim = 256

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        total_cond_dim = cond_dim + diffusion_step_embed_dim

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        for i in range(len(all_dims) - 1):
            self.down_blocks.append(
                ConditionalResBlock1D(all_dims[i], all_dims[i + 1], total_cond_dim,
                                     kernel_size, n_groups, cond_predict_scale)
            )

        for i in range(len(all_dims) - 1, 0, -1):
            self.up_blocks.append(
                ConditionalResBlock1D(all_dims[i] * 2, all_dims[i - 1], total_cond_dim,
                                     kernel_size, n_groups, cond_predict_scale)
            )

        self.mid_block = ConditionalResBlock1D(
            all_dims[-1], all_dims[-1], total_cond_dim, kernel_size, n_groups, cond_predict_scale
        )

        self.final_conv = nn.Sequential(nn.Conv1d(input_dim, input_dim, 1))

    def forward(self, x, timestep, cond):
        """
        x:         (B, action_horizon, action_dim)
        timestep:  (B,) float, scaled to [0, 1000]
        cond:      (B, cond_dim)
        """
        x = x.permute(0, 2, 1)
        t_emb = self.diffusion_step_encoder(timestep)
        global_cond = torch.cat([cond, t_emb], dim=-1)

        h = []
        for block in self.down_blocks:
            x = block(x, global_cond)
            h.append(x)

        x = self.mid_block(x, global_cond)

        for block in self.up_blocks:
            x = torch.cat([x, h.pop()], dim=1)
            x = block(x, global_cond)

        x = self.final_conv(x)
        return x.permute(0, 2, 1)


# ------------------------------------------------------------------
# Flow Matching Policy
# ------------------------------------------------------------------

class FlowPolicy(nn.Module):
    """Flow Matching policy.

    Training:  linear interpolation between x0 (Gaussian) and x1 (action).
    Inference: Euler integration of the learned vector field.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.obs_horizon = cfg["data"]["obs_horizon"]
        self.action_horizon = cfg["data"]["action_horizon"]
        self.action_dim = cfg["data"]["action_dim"]

        self.vision_encoder = VisionEncoder(cfg["vision_encoder"])
        vis_dim = self.vision_encoder.output_dim

        low_dim_keys = [k for k in cfg["data"]["obs_keys"] if "image" not in k]
        self.low_dim_keys = low_dim_keys
        self.low_dim_size = 0
        dim_map = {
            "robot0_eef_pos": 3, "robot0_eef_quat": 4, "robot0_gripper_qpos": 2,
            "robot0_joint_pos": 7, "ee_pos": 3, "ee_ori": 3,
            "ee_states": 6, "gripper_states": 2, "joint_states": 7,
        }
        for k in low_dim_keys:
            self.low_dim_size += dim_map.get(k, 3)

        n_vis_streams = 1
        if cfg["data"].get("use_eye_in_hand", False):
            self.eye_encoder = VisionEncoder(cfg["vision_encoder"])
            n_vis_streams = 2
        else:
            self.eye_encoder = None

        self.task_emb_dim = cfg.get("model", {}).get("task_emb_dim", 0)
        if self.task_emb_dim > 0:
            self.task_proj = nn.Linear(512, self.task_emb_dim)
        else:
            self.task_proj = None

        obs_feat_dim = (vis_dim * n_vis_streams + self.low_dim_size) * self.obs_horizon
        obs_feat_dim += self.task_emb_dim

        unet_cfg = cfg["unet"]
        self.vector_field_net = ConditionalUNet1D(
            input_dim=self.action_dim,
            cond_dim=obs_feat_dim,
            down_dims=unet_cfg["down_dims"],
            kernel_size=unet_cfg.get("kernel_size", 5),
            n_groups=unet_cfg.get("n_groups", 8),
            cond_predict_scale=unet_cfg.get("cond_predict_scale", False),
        )

        fm_cfg = cfg.get("flow_matching", cfg.get("diffusion", {}))
        self.num_flow_steps = fm_cfg.get("num_flow_steps", 10)

    def encode_obs(self, batch):
        features = []

        if "obs_agentview_image" in batch:
            vis_feat = self.vision_encoder(batch["obs_agentview_image"])
            features.append(vis_feat)

        if self.eye_encoder is not None and "obs_eye_in_hand_image" in batch:
            eye_feat = self.eye_encoder(batch["obs_eye_in_hand_image"])
            features.append(eye_feat)

        low_dim_parts = []
        for k in self.low_dim_keys:
            obs_key = f"obs_{k}"
            if obs_key in batch:
                low_dim_parts.append(batch[obs_key])
        if low_dim_parts:
            low_dim = torch.cat(low_dim_parts, dim=-1)
            features.append(low_dim)

        obs_cond = torch.cat(features, dim=-1)
        obs_cond = obs_cond.flatten(start_dim=1)

        if self.task_proj is not None and "task_emb" in batch:
            task_feat = self.task_proj(batch["task_emb"].to(obs_cond.device))
            obs_cond = torch.cat([obs_cond, task_feat], dim=-1)

        return obs_cond

    def compute_loss(self, batch):
        """FM training loss: MSE between predicted and target velocity."""
        x1 = batch["action"]   # demonstration actions, shape (B, Ta, Da)
        B = x1.shape[0]

        obs_cond = self.encode_obs(batch)

        x0 = torch.randn_like(x1)                      # Gaussian noise
        t = torch.rand(B, device=x1.device)            # t ~ Uniform[0, 1]
        t_b = t.view(-1, 1, 1)

        x_t = (1 - t_b) * x0 + t_b * x1               # linear interpolation
        target_v = x1 - x0                             # constant velocity field

        t_scaled = t * 1000                            # scale for SinusoidalPosEmb
        v_pred = self.vector_field_net(x_t, t_scaled, obs_cond)

        return F.mse_loss(v_pred, target_v)

    @torch.no_grad()
    def predict_action(self, batch):
        """Euler integration of the learned vector field."""
        obs_cond = self.encode_obs(batch)
        B = obs_cond.shape[0]

        x = torch.randn(B, self.action_horizon, self.action_dim, device=obs_cond.device)
        dt = 1.0 / self.num_flow_steps

        for i in range(self.num_flow_steps):
            t = torch.full((B,), i * dt, device=obs_cond.device)
            t_scaled = t * 1000
            v = self.vector_field_net(x, t_scaled, obs_cond)
            x = x + v * dt

        return x

    def sample_action(self, batch):
        """Alias for predict_action (for compatibility with rollout_evaluator)."""
        return self.predict_action(batch)


# ------------------------------------------------------------------
# EMA  (unchanged from diffusion_policy.py)
# ------------------------------------------------------------------

class EMAModel:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ema_p, model_p in zip(self.model.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
