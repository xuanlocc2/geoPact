from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_ch)
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + identity)


class ResidualBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(out_ch)
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + identity)


def match_spatial(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    diff = [ref.shape[i] - x.shape[i] for i in range(2, 5)]
    if all(delta == 0 for delta in diff):
        return x

    pads = []
    for delta in reversed(diff):
        if delta >= 0:
            before = delta // 2
            after = delta - before
            pads.extend([before, after])
        else:
            pads.extend([0, 0])

    if any(delta > 0 for delta in diff):
        x = F.pad(x, pads)

    slices = [slice(None), slice(None)]
    for axis, delta in enumerate(diff, start=2):
        if delta < 0:
            trim = -delta
            start = trim // 2
            end = start + ref.shape[axis]
            slices.append(slice(start, end))
        else:
            slices.append(slice(None))
    return x[tuple(slices)]


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = ResidualBlock(in_ch, out_ch)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        feat = self.block(x)
        return feat, self.pool(feat)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = ResidualBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        skip = match_spatial(skip, x)
        return self.block(torch.cat([x, skip], dim=1))


class TopogramEncoder2D(nn.Module):
    """Encode the physical x-z topogram plane without 3D ray-axis convolutions."""

    def __init__(self, in_channels: int, channels: tuple[int, int, int, int, int]):
        super().__init__()
        c1, c2, c3, c4, c5 = channels
        self.block1 = ResidualBlock2D(in_channels, c1)
        self.block2 = ResidualBlock2D(c1, c2)
        self.block3 = ResidualBlock2D(c2, c3)
        self.block4 = ResidualBlock2D(c3, c4)
        self.bottleneck = ResidualBlock2D(c4, c5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        t1 = self.block1(x)
        t2 = self.block2(self.pool(t1))
        t3 = self.block3(self.pool(t2))
        t4 = self.block4(self.pool(t3))
        tb = self.bottleneck(self.pool(t4))
        return t1, t2, t3, t4, tb


class TopogramGatedFusion(nn.Module):
    """Lift 2D topogram features along y and add a ControlNet-style residual."""

    def __init__(self, channels: int, gate_init_bias: float = -2.0):
        super().__init__()
        self.topo_projection = nn.Conv3d(channels, channels, kernel_size=1)
        self.zero_conv = nn.Conv3d(channels, channels, kernel_size=1)
        self.gate = nn.Conv3d(channels * 3, channels, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_init_bias)

    def forward(self, geometry: torch.Tensor, topogram_2d: torch.Tensor) -> torch.Tensor:
        target_x, target_y, target_z = geometry.shape[2:]
        if tuple(topogram_2d.shape[-2:]) != (target_x, target_z):
            topogram_2d = F.interpolate(topogram_2d, size=(target_x, target_z), mode="bilinear", align_corners=False)

        lifted = topogram_2d.unsqueeze(3).expand(-1, -1, -1, target_y, -1)
        topo = self.topo_projection(lifted)
        gate_input = torch.cat([geometry, topo, torch.abs(geometry - topo)], dim=1)
        gate = torch.sigmoid(self.gate(gate_input))
        return geometry + gate * self.zero_conv(topo)


class TopogramConditionedResidual(nn.Module):
    """Y-position-aware topogram residual fusion.

    The 2D scout is still encoded in its physical x-z plane. This module only
    adds 3D conditioning after the topogram feature has been lifted along y.
    It uses a small Fourier embedding of the anterior-posterior y coordinate
    and starts with a small non-zero residual instead of a zero residual.
    """

    def __init__(
        self,
        channels: int,
        coord_channels: int,
        gate_init_bias: float = -1.0,
        coord_fourier_bands: int = 3,
        coord_embedding_channels: int = 8,
        residual_init_scale: float = 0.01,
    ):
        super().__init__()
        if coord_channels <= 0:
            raise ValueError(f"coord_channels must be positive, got {coord_channels}")
        if coord_fourier_bands <= 0:
            raise ValueError(f"coord_fourier_bands must be positive, got {coord_fourier_bands}")
        if coord_embedding_channels <= 0:
            raise ValueError(f"coord_embedding_channels must be positive, got {coord_embedding_channels}")
        self.coord_fourier_bands = int(coord_fourier_bands)
        frequencies = torch.tensor([2.0**idx for idx in range(self.coord_fourier_bands)], dtype=torch.float32)
        self.register_buffer("coord_frequencies", frequencies.view(1, -1, 1, 1, 1), persistent=False)
        self.coord_projection = nn.Conv3d(self.coord_fourier_bands * 2, coord_embedding_channels, kernel_size=1)
        condition_channels = channels * 3 + coord_embedding_channels
        self.condition_net = nn.Sequential(
            nn.Conv3d(condition_channels, channels, kernel_size=1),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.residual_out = nn.Conv3d(channels, channels, kernel_size=1)
        self.gate_out = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.kaiming_normal_(self.residual_out.weight, nonlinearity="linear")
        self.residual_out.weight.data.mul_(float(residual_init_scale))
        nn.init.zeros_(self.residual_out.bias)
        nn.init.zeros_(self.gate_out.weight)
        nn.init.constant_(self.gate_out.bias, gate_init_bias)

    def _coord_embedding(self, coords_3d: torch.Tensor) -> torch.Tensor:
        y_coord = coords_3d[:, :1]
        angles = torch.pi * self.coord_frequencies.to(dtype=y_coord.dtype, device=y_coord.device) * y_coord
        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        return self.coord_projection(pe)

    def forward(self, geometry: torch.Tensor, topogram_2d: torch.Tensor, coords_3d: torch.Tensor) -> torch.Tensor:
        target_x, target_y, target_z = geometry.shape[2:]
        if tuple(topogram_2d.shape[-2:]) != (target_x, target_z):
            topogram_2d = F.interpolate(topogram_2d, size=(target_x, target_z), mode="bilinear", align_corners=False)

        lifted = topogram_2d.unsqueeze(3).expand(-1, -1, -1, target_y, -1)
        if coords_3d.shape[2:] != geometry.shape[2:]:
            coords_3d = F.interpolate(coords_3d, size=geometry.shape[2:], mode="trilinear", align_corners=False)
        coords_3d = coords_3d.to(dtype=geometry.dtype, device=geometry.device)

        coord_embedding = self._coord_embedding(coords_3d)
        condition = torch.cat([geometry, lifted, torch.abs(geometry - lifted), coord_embedding], dim=1)
        hidden = self.condition_net(condition)
        residual = self.residual_out(hidden)
        gate = torch.sigmoid(self.gate_out(hidden))
        return geometry + gate * residual


class MRIEncoder3D(nn.Module):
    """Encode Dixon MRI separately before it conditions the geometry stream."""

    def __init__(self, in_channels: int, base_channels: int, dropout: float):
        super().__init__()
        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        c5 = c4 * 2
        self.enc1 = EncoderBlock(in_channels, c1)
        self.enc2 = EncoderBlock(c1, c2)
        self.enc3 = EncoderBlock(c2, c3)
        self.enc4 = EncoderBlock(c3, c4)
        self.bottleneck = nn.Sequential(ResidualBlock(c4, c5), nn.Dropout3d(dropout))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)
        b = self.bottleneck(p4)
        return s1, s2, s3, s4, b


class VolumetricGatedFusion(nn.Module):
    """Project a 3D conditioning branch into geometry with a zero-initialized residual."""

    def __init__(self, geometry_channels: int, condition_channels: int, gate_init_bias: float = -2.0):
        super().__init__()
        self.condition_projection = nn.Conv3d(condition_channels, geometry_channels, kernel_size=1)
        self.zero_conv = nn.Conv3d(geometry_channels, geometry_channels, kernel_size=1)
        self.gate = nn.Conv3d(geometry_channels * 3, geometry_channels, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_init_bias)

    def forward(self, geometry: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if condition.shape[2:] != geometry.shape[2:]:
            condition = F.interpolate(condition, size=geometry.shape[2:], mode="trilinear", align_corners=False)
        projected = self.condition_projection(condition)
        gate_input = torch.cat([geometry, projected, torch.abs(geometry - projected)], dim=1)
        gate = torch.sigmoid(self.gate(gate_input))
        return geometry + gate * self.zero_conv(projected)


class WholeBodyContextEncoder(nn.Module):
    """Memory-conscious whole-body encoder with a low-resolution CT head.

    The branch sees a compact full-subject overview. Its pooled bottleneck
    conditions the local geometry stream, while a one-channel feature pyramid
    predicts an auxiliary whole-body CT without a memory-heavy 3D decoder.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        local_s4_channels: int,
        local_bottleneck_channels: int,
        dropout: float = 0.1,
        film_strength: float = 0.1,
        checkpoint_blocks: bool = True,
    ):
        super().__init__()
        c1 = int(base_channels)
        c2, c3, c4, c5 = c1 * 2, c1 * 4, c1 * 8, c1 * 16
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self.film_strength = float(film_strength)
        self.enc1 = EncoderBlock(in_channels, c1)
        self.enc2 = EncoderBlock(c1, c2)
        self.enc3 = EncoderBlock(c2, c3)
        self.enc4 = EncoderBlock(c3, c4)
        self.bottleneck = nn.Sequential(ResidualBlock(c4, c5), nn.Dropout3d(dropout))

        self.pyramid_heads = nn.ModuleList(
            [nn.Conv3d(channels, 1, kernel_size=1) for channels in (c1, c2, c3, c4, c5)]
        )
        self.ct_refine = nn.Sequential(
            nn.Conv3d(1, c1, kernel_size=3, padding=1),
            nn.InstanceNorm3d(c1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(c1, 1, kernel_size=1),
        )
        self.film_s4 = nn.Linear(c5, local_s4_channels * 2)
        self.film_bottleneck = nn.Linear(c5, local_bottleneck_channels * 2)
        nn.init.zeros_(self.film_s4.weight)
        nn.init.zeros_(self.film_s4.bias)
        nn.init.zeros_(self.film_bottleneck.weight)
        nn.init.zeros_(self.film_bottleneck.bias)

    def _run(self, module: nn.Module, x: torch.Tensor):
        if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    @staticmethod
    def _upsample_scalar(x: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
        if x.shape[2:] == target_shape:
            return x
        return F.interpolate(x, size=target_shape, mode="trilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"Expected global input [B,C,X,Y,Z], got {tuple(x.shape)}")
        g1, p1 = self._run(self.enc1, x)
        g2, p2 = self._run(self.enc2, p1)
        g3, p3 = self._run(self.enc3, p2)
        g4, p4 = self._run(self.enc4, p3)
        gb = self._run(self.bottleneck, p4)

        target_shape = tuple(int(value) for value in x.shape[2:])
        pyramid = x.new_zeros((x.shape[0], 1, *target_shape))
        for feature, head in zip((g1, g2, g3, g4, gb), self.pyramid_heads):
            pyramid = pyramid + self._upsample_scalar(head(feature), target_shape)
        global_ct = self.ct_refine(pyramid)

        descriptor = F.adaptive_avg_pool3d(gb, output_size=1).flatten(1)
        return {
            "global_ct": global_ct,
            "film_s4": self.film_s4(descriptor),
            "film_bottleneck": self.film_bottleneck(descriptor),
        }


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.2,
        metadata_dim: int = 0,
        use_metadata_film: bool = False,
    ):
        super().__init__()
        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        c5 = c4 * 2
        self.use_metadata_film = bool(use_metadata_film and metadata_dim > 0)

        self.enc1 = EncoderBlock(in_channels, c1)
        self.enc2 = EncoderBlock(c1, c2)
        self.enc3 = EncoderBlock(c2, c3)
        self.enc4 = EncoderBlock(c3, c4)
        self.bottleneck = nn.Sequential(ResidualBlock(c4, c5), nn.Dropout3d(dropout))
        if self.use_metadata_film:
            self.metadata_mlp = nn.Sequential(
                nn.Linear(metadata_dim, c5),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Linear(c5, c5 * 2),
            )
        else:
            self.metadata_mlp = None
        self.dec4 = DecoderBlock(c5, c4, c4)
        self.dec3 = DecoderBlock(c4, c3, c3)
        self.dec2 = DecoderBlock(c3, c2, c2)
        self.dec1 = DecoderBlock(c2, c1, c1)
        self.out_conv = nn.Conv3d(c1, out_channels, kernel_size=1)

    def _apply_metadata_film(self, x: torch.Tensor, metadata: torch.Tensor | None) -> torch.Tensor:
        if not self.use_metadata_film:
            return x
        if metadata is None:
            metadata = torch.zeros((x.shape[0], self.metadata_mlp[0].in_features), dtype=x.dtype, device=x.device)
        metadata = metadata.to(device=x.device, dtype=x.dtype)
        gamma_beta = self.metadata_mlp(metadata)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=1)
        gamma = gamma.view(gamma.shape[0], gamma.shape[1], 1, 1, 1)
        beta = beta.view(beta.shape[0], beta.shape[1], 1, 1, 1)
        return x * (1.0 + gamma) + beta

    def forward(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor | None = None,
        topogram: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del topogram
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)
        b = self._apply_metadata_film(self.bottleneck(p4), metadata)
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return self.out_conv(d1)


class GatedTopogramUNet3D(UNet3D):
    """Geometry U-Net with a 2D topogram branch and optional gated 3D MRI branch."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.2,
        metadata_dim: int = 0,
        use_metadata_film: bool = False,
        geometry_in_channels: int = 2,
        geometry_channel_indices: tuple[int, ...] | list[int] | None = None,
        topogram_channel_index: int = 2,
        topogram_gate_init_bias: float = -2.0,
        topogram_conditioned: dict | None = None,
        mri_channel_indices: tuple[int, ...] | list[int] | None = None,
        mri_base_channels: int | None = None,
        mri_gate_init_bias: float = -2.0,
        global_context: dict | None = None,
    ):
        if not 0 <= int(topogram_channel_index) < int(in_channels):
            raise ValueError(f"Invalid topogram channel index {topogram_channel_index} for {in_channels} input channels")
        mri_channel_indices = tuple(int(idx) for idx in (mri_channel_indices or ()))
        if len(set(mri_channel_indices)) != len(mri_channel_indices):
            raise ValueError("MRI channel indices must be unique")
        if int(topogram_channel_index) in mri_channel_indices:
            raise ValueError("Topogram channel cannot also be an MRI channel")
        if any(idx < 0 or idx >= int(in_channels) for idx in mri_channel_indices):
            raise ValueError(f"Invalid MRI channel index in {mri_channel_indices} for {in_channels} input channels")

        topogram_conditioned = dict(topogram_conditioned or {})
        self.topogram_conditioned_enabled = bool(topogram_conditioned.get("enabled", False))
        topogram_conditioned_coord_indices = tuple(
            int(idx) for idx in topogram_conditioned.get("coord_channel_indices", ())
        )
        if any(idx < 0 or idx >= int(in_channels) for idx in topogram_conditioned_coord_indices):
            raise ValueError(
                f"Invalid topogram_conditioned.coord_channel_indices={topogram_conditioned_coord_indices} "
                f"for {in_channels} input channels"
            )
        if int(topogram_channel_index) in topogram_conditioned_coord_indices:
            raise ValueError("Topogram channel cannot also be a topogram-conditioned coordinate channel")
        if set(mri_channel_indices) & set(topogram_conditioned_coord_indices):
            raise ValueError("MRI channels cannot also be topogram-conditioned coordinate channels")

        if geometry_channel_indices is None:
            condition_indices = {int(topogram_channel_index), *mri_channel_indices, *topogram_conditioned_coord_indices}
            geometry_channel_indices = tuple(idx for idx in range(int(in_channels)) if idx not in condition_indices)
        else:
            geometry_channel_indices = tuple(int(idx) for idx in geometry_channel_indices)
        if len(set(geometry_channel_indices)) != len(geometry_channel_indices):
            raise ValueError("Geometry channel indices must be unique")
        if any(idx < 0 or idx >= int(in_channels) for idx in geometry_channel_indices):
            raise ValueError(f"Invalid geometry_channel_indices={geometry_channel_indices} for {in_channels} input channels")
        if int(topogram_channel_index) in geometry_channel_indices:
            raise ValueError("Topogram channel cannot also be a geometry channel")
        if set(mri_channel_indices) & set(geometry_channel_indices):
            raise ValueError("MRI channels cannot also be geometry channels")
        if len(geometry_channel_indices) != int(geometry_in_channels):
            raise ValueError(
                f"geometry_in_channels={geometry_in_channels} but geometry_channel_indices={geometry_channel_indices}"
            )

        super().__init__(
            in_channels=geometry_in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            dropout=dropout,
            metadata_dim=metadata_dim,
            use_metadata_film=use_metadata_film,
        )
        self.input_channels = int(in_channels)
        self.geometry_in_channels = int(geometry_in_channels)
        self.geometry_channel_indices = tuple(geometry_channel_indices)
        self.topogram_channel_index = int(topogram_channel_index)
        self.mri_channel_indices = mri_channel_indices
        self.topogram_conditioned_coord_indices = topogram_conditioned_coord_indices
        channels = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16)
        self.topogram_encoder = TopogramEncoder2D(1, channels)
        self.fuse1 = TopogramGatedFusion(channels[0], topogram_gate_init_bias)
        self.fuse2 = TopogramGatedFusion(channels[1], topogram_gate_init_bias)
        self.fuse3 = TopogramGatedFusion(channels[2], topogram_gate_init_bias)
        self.fuse4 = TopogramGatedFusion(channels[3], topogram_gate_init_bias)
        self.fuse_bottleneck = TopogramGatedFusion(channels[4], topogram_gate_init_bias)
        conditioned_scales = set(topogram_conditioned.get("scales", ("s4", "bottleneck")))
        conditioned_gate_bias = float(topogram_conditioned.get("gate_bias", topogram_gate_init_bias))
        conditioned_coord_fourier_bands = int(topogram_conditioned.get("coord_fourier_bands", 3))
        conditioned_coord_embedding_channels = int(topogram_conditioned.get("coord_embedding_channels", 8))
        conditioned_residual_init_scale = float(topogram_conditioned.get("residual_init_scale", 0.01))
        conditioned_coord_count = len(self.topogram_conditioned_coord_indices)
        self.topogram_conditioned_scales = conditioned_scales if self.topogram_conditioned_enabled else set()
        if self.topogram_conditioned_enabled and conditioned_coord_count == 0:
            raise ValueError("topogram_conditioned.enabled requires coord_channel_indices")
        self.topogram_conditioned3 = (
            TopogramConditionedResidual(
                channels[2],
                conditioned_coord_count,
                gate_init_bias=conditioned_gate_bias,
                coord_fourier_bands=conditioned_coord_fourier_bands,
                coord_embedding_channels=conditioned_coord_embedding_channels,
                residual_init_scale=conditioned_residual_init_scale,
            )
            if "s3" in self.topogram_conditioned_scales
            else None
        )
        self.topogram_conditioned4 = (
            TopogramConditionedResidual(
                channels[3],
                conditioned_coord_count,
                gate_init_bias=conditioned_gate_bias,
                coord_fourier_bands=conditioned_coord_fourier_bands,
                coord_embedding_channels=conditioned_coord_embedding_channels,
                residual_init_scale=conditioned_residual_init_scale,
            )
            if "s4" in self.topogram_conditioned_scales
            else None
        )
        self.topogram_conditioned_bottleneck = (
            TopogramConditionedResidual(
                channels[4],
                conditioned_coord_count,
                gate_init_bias=conditioned_gate_bias,
                coord_fourier_bands=conditioned_coord_fourier_bands,
                coord_embedding_channels=conditioned_coord_embedding_channels,
                residual_init_scale=conditioned_residual_init_scale,
            )
            if "bottleneck" in self.topogram_conditioned_scales
            else None
        )
        if self.topogram_conditioned3 is not None:
            self.fuse3 = None
        if self.topogram_conditioned4 is not None:
            self.fuse4 = None
        if self.topogram_conditioned_bottleneck is not None:
            self.fuse_bottleneck = None
        if self.mri_channel_indices:
            mri_base_channels = int(mri_base_channels if mri_base_channels is not None else base_channels)
            if mri_base_channels <= 0:
                raise ValueError(f"mri_base_channels must be positive, got {mri_base_channels}")
            mri_channels = (
                mri_base_channels,
                mri_base_channels * 2,
                mri_base_channels * 4,
                mri_base_channels * 8,
                mri_base_channels * 16,
            )
            self.mri_encoder = MRIEncoder3D(len(self.mri_channel_indices), mri_base_channels, dropout)
            self.mri_fuse1 = VolumetricGatedFusion(channels[0], mri_channels[0], mri_gate_init_bias)
            self.mri_fuse2 = VolumetricGatedFusion(channels[1], mri_channels[1], mri_gate_init_bias)
            self.mri_fuse3 = VolumetricGatedFusion(channels[2], mri_channels[2], mri_gate_init_bias)
            self.mri_fuse4 = VolumetricGatedFusion(channels[3], mri_channels[3], mri_gate_init_bias)
            self.mri_fuse_bottleneck = VolumetricGatedFusion(channels[4], mri_channels[4], mri_gate_init_bias)
        else:
            self.mri_encoder = None

        global_context = dict(global_context or {})
        self.global_context_enabled = bool(global_context.get("enabled", False))
        if self.global_context_enabled:
            global_in_channels = int(global_context.get("in_channels", 3))
            global_base_channels = int(global_context.get("base_channels", 8))
            if global_in_channels <= 0 or global_base_channels <= 0:
                raise ValueError("global_context in_channels/base_channels must be positive")
            self.global_context_encoder = WholeBodyContextEncoder(
                in_channels=global_in_channels,
                base_channels=global_base_channels,
                local_s4_channels=channels[3],
                local_bottleneck_channels=channels[4],
                dropout=float(global_context.get("dropout", 0.1)),
                film_strength=float(global_context.get("film_strength", 0.1)),
                checkpoint_blocks=bool(global_context.get("checkpoint_blocks", True)),
            )
        else:
            self.global_context_encoder = None

    def prepare_global_context(self, global_input: torch.Tensor | None) -> dict[str, torch.Tensor] | None:
        if not self.global_context_enabled:
            return None
        if global_input is None:
            raise ValueError("global_context.enabled requires batch['global_input']")
        return self.global_context_encoder(global_input)

    def _apply_global_film(self, feature: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        if parameters.shape[0] != feature.shape[0]:
            if parameters.shape[0] != 1:
                raise ValueError(
                    f"Global context batch {parameters.shape[0]} cannot condition local batch {feature.shape[0]}"
                )
            parameters = parameters.expand(feature.shape[0], -1)
        gamma, beta = torch.chunk(parameters.to(dtype=feature.dtype, device=feature.device), 2, dim=1)
        gamma = gamma.view(gamma.shape[0], gamma.shape[1], 1, 1, 1)
        beta = beta.view(beta.shape[0], beta.shape[1], 1, 1, 1)
        strength = self.global_context_encoder.film_strength
        return feature * (1.0 + strength * torch.tanh(gamma)) + strength * beta

    def _split_inputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if x.ndim != 5 or x.shape[1] != self.input_channels:
            raise ValueError(f"Expected [B, {self.input_channels}, X, Y, Z], got {tuple(x.shape)}")
        geometry = x[:, list(self.geometry_channel_indices)]
        # The transform broadcasts the x-z scout along y only for synchronized
        # sliding-window crops. Collapse it before the dedicated 2D encoder.
        topogram_2d = x[:, self.topogram_channel_index : self.topogram_channel_index + 1].mean(dim=3)
        mri = x[:, list(self.mri_channel_indices)] if self.mri_channel_indices else None
        topogram_coords = (
            x[:, list(self.topogram_conditioned_coord_indices)] if self.topogram_conditioned_coord_indices else None
        )
        return geometry, topogram_2d, mri, topogram_coords

    def forward(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor | None = None,
        topogram: torch.Tensor | None = None,
        global_input: torch.Tensor | None = None,
        global_context: dict[str, torch.Tensor] | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        del topogram
        if self.global_context_enabled and global_context is None:
            global_context = self.prepare_global_context(global_input)
        geometry, topogram_2d, mri, topogram_coords = self._split_inputs(x)
        t1, t2, t3, t4, tb = self.topogram_encoder(topogram_2d)
        mri_features = self.mri_encoder(mri) if mri is not None else None

        s1, _ = self.enc1(geometry)
        s1 = self.fuse1(s1, t1)
        if mri_features is not None:
            s1 = self.mri_fuse1(s1, mri_features[0])
        p1 = self.enc1.pool(s1)

        s2, _ = self.enc2(p1)
        s2 = self.fuse2(s2, t2)
        if mri_features is not None:
            s2 = self.mri_fuse2(s2, mri_features[1])
        p2 = self.enc2.pool(s2)

        s3, _ = self.enc3(p2)
        if self.topogram_conditioned3 is not None:
            s3 = self.topogram_conditioned3(s3, t3, topogram_coords)
        else:
            s3 = self.fuse3(s3, t3)
        if mri_features is not None:
            s3 = self.mri_fuse3(s3, mri_features[2])
        p3 = self.enc3.pool(s3)

        s4, _ = self.enc4(p3)
        if self.topogram_conditioned4 is not None:
            s4 = self.topogram_conditioned4(s4, t4, topogram_coords)
        else:
            s4 = self.fuse4(s4, t4)
        if mri_features is not None:
            s4 = self.mri_fuse4(s4, mri_features[3])
        if global_context is not None:
            s4 = self._apply_global_film(s4, global_context["film_s4"])
        p4 = self.enc4.pool(s4)

        b = self.bottleneck(p4)
        if self.topogram_conditioned_bottleneck is not None:
            b = self.topogram_conditioned_bottleneck(b, tb, topogram_coords)
        else:
            b = self.fuse_bottleneck(b, tb)
        if mri_features is not None:
            b = self.mri_fuse_bottleneck(b, mri_features[4])
        if global_context is not None:
            b = self._apply_global_film(b, global_context["film_bottleneck"])
        b = self._apply_metadata_film(b, metadata)
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        ct = self.out_conv(d1)
        if return_aux:
            output = {"ct": ct}
            if global_context is not None and "global_ct" in global_context:
                output["global_ct"] = global_context["global_ct"]
            return output
        return ct


def build_model(
    in_channels: int = 3,
    out_channels: int = 1,
    base_channels: int = 32,
    dropout: float = 0.2,
    metadata_dim: int = 0,
    use_metadata_film: bool = False,
    model_name: str = "mdixon_resunet3d",
    geometry_in_channels: int | None = None,
    geometry_channel_indices: tuple[int, ...] | list[int] | None = None,
    topogram_channel_index: int = -1,
    topogram_gate_init_bias: float = -2.0,
    topogram_conditioned: dict | None = None,
    mri_channel_indices: tuple[int, ...] | list[int] | None = None,
    mri_base_channels: int | None = None,
    mri_gate_init_bias: float = -2.0,
    global_context: dict | None = None,
) -> nn.Module:
    common = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        dropout=dropout,
        metadata_dim=metadata_dim,
        use_metadata_film=use_metadata_film,
    )
    if model_name in {"mdixon_resunet3d", "resunet3d", "unet3d"}:
        return UNet3D(**common)
    if model_name in {
        "gated_topogram_2d_resunet3d",
        "topogram_2d_gated",
        "gated_topogram_mri_resunet3d",
        "topogram_mri_2d_gated",
        "gated_topogram_conditioned_mri_resunet3d",
        "topogram_conditioned_mri_2d_gated",
    }:
        return GatedTopogramUNet3D(
            **common,
            geometry_in_channels=geometry_in_channels if geometry_in_channels is not None else in_channels - 1,
            geometry_channel_indices=geometry_channel_indices,
            topogram_channel_index=topogram_channel_index if topogram_channel_index >= 0 else in_channels - 1,
            topogram_gate_init_bias=topogram_gate_init_bias,
            topogram_conditioned=topogram_conditioned,
            mri_channel_indices=mri_channel_indices,
            mri_base_channels=mri_base_channels,
            mri_gate_init_bias=mri_gate_init_bias,
            global_context=global_context,
        )
    raise ValueError(f"Unknown model.name: {model_name}")
