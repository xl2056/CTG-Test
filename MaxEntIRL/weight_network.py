"""
Context-conditional weight network for MaxEnt IRL.

Implements r_theta(s, a, c) = w_theta(c)^T . phi(s, a), where:
  - phi(s, a) is the existing 7-dim hand-crafted feature vector
  - c is an encoded context (local map raster + neighbor trajectories + ego history)
  - w_theta(c) is a small neural network that outputs per-scene feature weights

The encoders are reused from tbsim.models (RasterizedMapEncoder, AgentHistoryEncoder,
NeighborHistoryEncoder) so behavior and normalization match the main diffusion model.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from tbsim.models.base_models import MLP, RasterizedMapEncoder
from tbsim.models.diffuser_helpers import AgentHistoryEncoder, NeighborHistoryEncoder


class WeightNetwork(nn.Module):
    """
    Context -> per-sample feature weights of shape (B, F).

    Forward takes a dict containing a subset of:
        image:                         (B, C, H, W)
        history_positions:             (B, T, 2)
        history_yaws:                  (B, T, 1)
        history_speeds:                (B, T)
        history_availabilities:        (B, T)
        extent:                        (B, 3)
        all_other_agents_history_positions:       (B, Q, T, 2)
        all_other_agents_history_yaws:            (B, Q, T, 1)
        all_other_agents_history_speeds:          (B, Q, T)
        all_other_agents_history_availabilities:  (B, Q, T)
        all_other_agents_extents:                 (B, Q, 3)

    Any stream that is missing from the context dict is disabled (its feature
    vector is replaced with zeros). The output weight is unconstrained; a
    `tanh * scale` activation is available via `weight_scale` to stabilize
    training.
    """

    def __init__(
        self,
        feature_dim: int = 7,
        map_feat_dim: int = 128,
        nbr_feat_dim: int = 128,
        ego_feat_dim: int = 64,
        mlp_hidden: Optional[List[int]] = None,
        num_history_steps: int = 21,
        map_channels: int = 3,
        map_image_hw: int = 224,
        map_arch: str = "resnet18",
        use_map: bool = True,
        use_neighbors: bool = True,
        use_ego: bool = True,
        weight_scale: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.use_map = use_map
        self.use_neighbors = use_neighbors
        self.use_ego = use_ego
        self.map_feat_dim = map_feat_dim if use_map else 0
        self.nbr_feat_dim = nbr_feat_dim if use_neighbors else 0
        self.ego_feat_dim = ego_feat_dim if use_ego else 0
        self.weight_scale = weight_scale

        if use_map:
            self.map_encoder = RasterizedMapEncoder(
                model_arch=map_arch,
                input_image_shape=(map_channels, map_image_hw, map_image_hw),
                feature_dim=map_feat_dim,
                output_activation=nn.ReLU,
            )
        else:
            self.map_encoder = None

        if use_ego:
            self.ego_encoder = AgentHistoryEncoder(
                num_steps=num_history_steps,
                out_dim=ego_feat_dim,
                use_norm=True,
            )
        else:
            self.ego_encoder = None

        if use_neighbors:
            self.nbr_encoder = NeighborHistoryEncoder(
                num_steps=num_history_steps,
                out_dim=nbr_feat_dim,
                use_norm=True,
            )
        else:
            self.nbr_encoder = None

        concat_dim = self.map_feat_dim + self.nbr_feat_dim + self.ego_feat_dim
        if concat_dim == 0:
            raise ValueError("WeightNetwork requires at least one of use_map, use_neighbors, use_ego to be True.")

        if mlp_hidden is None:
            mlp_hidden = [128]
        self.head = MLP(
            input_dim=concat_dim,
            output_dim=feature_dim,
            layer_dims=tuple(mlp_hidden),
            normalization=True,
            output_activation=None,
        )

    @staticmethod
    def _has(ctx: Dict[str, torch.Tensor], keys: List[str]) -> bool:
        return all((k in ctx) and (ctx[k] is not None) for k in keys)

    def encode_context(self, context: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return (B, concat_dim) embedding for a batched context dict."""
        device = next(self.parameters()).device
        feats: List[torch.Tensor] = []
        batch_size: Optional[int] = None

        if self.use_map:
            if self._has(context, ["image"]):
                img = context["image"].to(device).float()
                # squeeze accidental singleton time dim
                if img.dim() == 5 and img.shape[1] == 1:
                    img = img.squeeze(1)
                map_feat = self.map_encoder(img)
                batch_size = map_feat.shape[0]
                feats.append(map_feat)
            else:
                feats.append(None)  # placeholder

        if self.use_ego:
            if self._has(context, ["history_positions", "history_yaws", "history_speeds", "extent"]):
                pos = context["history_positions"].to(device).float()
                yaw = context["history_yaws"].to(device).float()
                speed = context["history_speeds"].to(device).float()
                extent = context["extent"].to(device).float()
                avail = context.get("history_availabilities")
                if avail is None:
                    avail = torch.ones_like(speed, dtype=torch.bool)
                # AgentHistoryEncoder.prepare_hist_in does `hist_in[~avail] = 0`,
                # which requires a boolean tensor. Features are saved as float.
                avail = avail.to(device).bool()
                ego_feat = self.ego_encoder(pos, yaw, speed, extent, avail)
                batch_size = ego_feat.shape[0] if batch_size is None else batch_size
                feats.append(ego_feat)
            else:
                feats.append(None)

        if self.use_neighbors:
            nbr_keys = [
                "all_other_agents_history_positions",
                "all_other_agents_history_yaws",
                "all_other_agents_history_speeds",
                "all_other_agents_extents",
            ]
            if self._has(context, nbr_keys):
                pos = context["all_other_agents_history_positions"].to(device).float()
                yaw = context["all_other_agents_history_yaws"].to(device).float()
                speed = context["all_other_agents_history_speeds"].to(device).float()
                extent = context["all_other_agents_extents"].to(device).float()
                avail = context.get("all_other_agents_history_availabilities")
                if avail is None:
                    avail = torch.ones_like(speed, dtype=torch.bool)
                # See ego branch: availability must be a bool tensor for
                # prepare_hist_in's `~avail` masking step.
                avail = avail.to(device).bool()
                nbr_feat = self.nbr_encoder(pos, yaw, speed, extent, avail)
                batch_size = nbr_feat.shape[0] if batch_size is None else batch_size
                feats.append(nbr_feat)
            else:
                feats.append(None)

        if batch_size is None:
            raise ValueError("WeightNetwork.encode_context received no usable context streams.")

        # Fill missing streams with zeros of matching shape
        resolved: List[torch.Tensor] = []
        stream_dims = []
        if self.use_map:
            stream_dims.append(self.map_feat_dim)
        if self.use_ego:
            stream_dims.append(self.ego_feat_dim)
        if self.use_neighbors:
            stream_dims.append(self.nbr_feat_dim)

        for feat, dim in zip(feats, stream_dims):
            if feat is None:
                resolved.append(torch.zeros((batch_size, dim), device=device))
            else:
                resolved.append(feat)

        return torch.cat(resolved, dim=-1)

    def forward(self, context: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        context: dict of torch tensors (see class docstring).
        Returns: (B, feature_dim) weight vectors.
        """
        embed = self.encode_context(context)
        w = self.head(embed)
        if self.weight_scale is not None:
            w = torch.tanh(w) * self.weight_scale
        return w


def build_weight_network_from_config(cfg) -> WeightNetwork:
    """Build a WeightNetwork from an irl_config.FeatureExtractionConfig-style object."""
    wn_cfg = getattr(cfg, "weight_network", None)
    if wn_cfg is None:
        wn_cfg = {}

    def _g(key, default):
        if isinstance(wn_cfg, dict):
            return wn_cfg.get(key, default)
        return getattr(wn_cfg, key, default)

    num_history_steps = _g("num_history_steps", None)
    if num_history_steps is None:
        num_history_steps = cfg.history_num_frames + 1

    return WeightNetwork(
        feature_dim=len(cfg.feature_names),
        map_feat_dim=_g("map_feat_dim", 128),
        nbr_feat_dim=_g("nbr_feat_dim", 128),
        ego_feat_dim=_g("ego_feat_dim", 64),
        mlp_hidden=_g("mlp_hidden", [128]),
        num_history_steps=num_history_steps,
        map_channels=_g("map_channels", 3),
        map_image_hw=_g("map_image_hw", 224),
        map_arch=_g("map_arch", "resnet18"),
        use_map=_g("use_map", True),
        use_neighbors=_g("use_neighbors", True),
        use_ego=_g("use_ego", True),
        weight_scale=_g("weight_scale", None),
    )
