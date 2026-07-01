from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class SpectrePatchConfig:
    patch_size: tuple[int, int, int] = (128, 128, 64)  # (H, W, D)
    grid_size: tuple[int, int, int] = (3, 3, 4)
    resize_mode: str = "trilinear"
    align_corners: bool = False

    @property
    def target_shape_hwd(self) -> tuple[int, int, int]:
        gh, gw, gd = self.grid_size
        ph, pw, pd = self.patch_size
        return gh * ph, gw * pw, gd * pd


class SpectreVolumeWrapper:
    def __init__(self, model: torch.nn.Module, config: SpectrePatchConfig | None = None):
        self.model = model
        self.config = config or SpectrePatchConfig()

    @staticmethod
    def _ensure_batch(volume: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if volume.ndim == 4:
            return volume.unsqueeze(0), True
        if volume.ndim == 5:
            return volume, False
        raise ValueError("volume must be [C,D,H,W] or [B,C,D,H,W]")

    def _to_hwd(self, volume_bcdhw: torch.Tensor) -> torch.Tensor:
        return volume_bcdhw.permute(0, 1, 3, 4, 2).contiguous()

    def _resize_to_target(self, volume_bchwd: torch.Tensor) -> torch.Tensor:
        target_hwd = self.config.target_shape_hwd
        current_hwd = tuple(int(v) for v in volume_bchwd.shape[-3:])
        if current_hwd == target_hwd:
            return volume_bchwd

        return F.interpolate(
            volume_bchwd,
            size=target_hwd,
            mode=self.config.resize_mode,
            align_corners=self.config.align_corners,
        )

    def volume_to_crops(self, volume: torch.Tensor) -> torch.Tensor:
        volume, _ = self._ensure_batch(volume)
        volume = self._to_hwd(volume)
        volume = self._resize_to_target(volume)

        b, c, h, w, d = volume.shape
        gh, gw, gd = self.config.grid_size
        ph, pw, pd = self.config.patch_size

        expected_hwd = self.config.target_shape_hwd
        if (h, w, d) != expected_hwd:
            raise ValueError(
                f"Unexpected spatial shape {(h, w, d)} after resize; expected {expected_hwd}"
            )

        x = volume.reshape(b, c, gh, ph, gw, pw, gd, pd)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x = x.view(b, gh * gw * gd, c, ph, pw, pd)
        return x

    def _model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _ensure_model_on_device(self, device: torch.device) -> None:
        if self._model_device() != device:
            self.model = self.model.to(device)
        self.model.eval()

    def infer_from_volume(
        self,
        volume: torch.Tensor,
        device: torch.device | str | None = None,
        use_amp: bool = True,
    ) -> torch.Tensor:
        target_device = torch.device(device) if device is not None else self._model_device()
        self._ensure_model_on_device(target_device)

        volume = volume.to(target_device, non_blocking=True)
        crops = self.volume_to_crops(volume)

        amp_enabled = use_amp and target_device.type == "cuda"
        with torch.inference_mode(), torch.autocast(
            device_type=target_device.type,
            enabled=amp_enabled,
        ):
            return self.model(crops, grid_size=self.config.grid_size)

    def infer_from_batch(self, batch: dict[str, torch.Tensor | Sequence[str]]) -> torch.Tensor:
        if "image" not in batch:
            raise KeyError("Expected batch dict with key 'image'")
        images = batch["image"]
        if not isinstance(images, torch.Tensor):
            raise TypeError("batch['image'] must be a torch.Tensor")
        return self.infer_from_volume(images)
