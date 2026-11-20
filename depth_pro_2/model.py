# Adapted from Apple Depth Pro 2. See LICENSE and UPSTREAM.md.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Iterator, Literal, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .network.decoder import DPTDecoder, create_inverse_depth_head
from .network.encoder import KVCache, MultiViewEncoder, autocast_for
from .network.fov import FOVEstimator
from .streaming import DepthStream, Prediction

INPUT_MULTIPLE = 32
VideoMode = Literal["buffered", "unbuffered"]
DEFAULT_WINDOW = 16
DEFAULT_OVERLAP = 4

@dataclass
class DepthPro2Config:
    backbone_preset: str
    reassembled_blocks: tuple[int, int, int, int]
    cross_frame_blocks: tuple[int, ...]
    dims_reassemble: tuple[int, int, int, int]
    dim_decoder: int



PRESETS: dict[str, DepthPro2Config] = {
    "depthpro2-s": DepthPro2Config(
        backbone_preset="vit_small_patch16_dinov3",
        reassembled_blocks=(2, 5, 8, 11),
        cross_frame_blocks=(5, 7, 9, 11),
        dims_reassemble=(48, 96, 192, 384),
        dim_decoder=64,
    ),
    "depthpro2-b": DepthPro2Config(
        backbone_preset="vit_base_patch16_dinov3",
        reassembled_blocks=(2, 5, 8, 11),
        cross_frame_blocks=(5, 7, 9, 11),
        dims_reassemble=(96, 192, 384, 768),
        dim_decoder=128,
    ),
    "depthpro2-l": DepthPro2Config(
        backbone_preset="vit_large_patch16_dinov3.lvd1689m",
        reassembled_blocks=(5, 11, 17, 23),
        cross_frame_blocks=(9, 11, 13, 15, 17, 19, 21, 23),
        dims_reassemble=(256, 512, 1024, 1024),
        dim_decoder=256,
    ),
    "depthpro2-h": DepthPro2Config(
        backbone_preset="vit_huge_plus_patch16_dinov3",
        reassembled_blocks=(7, 15, 23, 31),
        cross_frame_blocks=(11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31),
        dims_reassemble=(256, 512, 1024, 1024),
        dim_decoder=256,
    ),
}


class DepthPro2(nn.Module):
    def __init__(
        self,
        backbone_preset: str,
        reassembled_blocks: Sequence[int],
        cross_frame_blocks: Sequence[int],
        dims_reassemble: Sequence[int],
        dim_decoder: int,
        fov: bool = True,
        precision: torch.dtype = torch.float32,
    ):
        """Initialize Depth Pro 2; ``precision`` is the autocast dtype, float32 disabling it."""
        super().__init__()
        self.encoder = MultiViewEncoder(
            backbone_preset=backbone_preset,
            reassembled_blocks=reassembled_blocks,
            cross_frame_blocks=cross_frame_blocks,
            dims_reassemble=dims_reassemble,
        )
        self.decoder = DPTDecoder(dims_reassemble=dims_reassemble, dim_decoder=dim_decoder)
        self.head = create_inverse_depth_head(self.decoder.dim_out)
        self.fov = FOVEstimator(precision) if fov else None
        self.precision = precision

    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        """Predict focal-normalized inverse depth ``(T, 1, H, W)`` for frames ``(T, 3, H, W)``.

        With a ``cache``, ``x`` is one frame ``(1, 3, H, W)``, attending the cached frames too.
        """
        with autocast_for(x.device, self.precision)():
            features, grid_size = self.encoder.encode(x, cache)

        inverse_depth = []
        for frame in range(x.shape[0]):
            tokens = [f[frame : frame + 1] for f in features]
            inverse_depth.append(self.decode_frame(tokens, grid_size))
        return torch.cat(inverse_depth)

    def decode_frame(
        self, features: Sequence[torch.Tensor], grid_size: tuple[int, int]
    ) -> torch.Tensor:
        """Decode one frame's tapped tokens into inverse depth ``(1, 1, H, W)``."""
        device = features[0].device
        with autocast_for(device, self.precision)():
            decoded = self.decoder(self.encoder.reassemble_tokens(features, grid_size))
        return self.head(decoded.float())

    def predict_inverse_depth(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        """Inverse depth ``(T, H, W)`` for frames ``(T, 3, H, W)`` at their native resolution."""
        height, width = x.shape[-2:]
        pad_h = -height % INPUT_MULTIPLE
        pad_w = -width % INPUT_MULTIPLE
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return self.forward(x, cache)[:, 0, :height, :width]

    def infer(
        self,
        x: torch.Tensor,
        f_px: float | torch.Tensor | None = None,
    ) -> Prediction:
        """Infer inverse depth, metric depth and focal length at the native input resolution.

        Args:
            x: Normalized image ``(3, H, W)`` or clip ``(T, 3, H, W)``. A clip's frames attend to
                each other and share one focal length, so pass unrelated photos one at a time.
            f_px: Focal length in pixels of ``x``, e.g. from EXIF; one estimate per clip by default.

        Returns:
            ``inverse_depth`` normalized by focal length and ``depth`` in meters along the optical
            axis, both ``(H, W)`` or ``(T, H, W)``; depth is infinite where infinitely far, such as
            sky. ``depth`` and ``focallength_px``, a 0-d tensor on the model's device, are ``None``
            if no focal length is given or estimated.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"infer takes a tensor from transform(image), got {type(x).__name__}.")
        if x.ndim not in (3, 4) or x.shape[-3] != 3 or not x.is_floating_point():
            raise ValueError(
                "infer takes a float (3, H, W) or (T, 3, H, W) tensor from transform(image), "
                f"got {x.dtype} {tuple(x.shape)}."
            )
        is_image = x.ndim == 3
        if is_image:
            x = x.unsqueeze(0)
        width = x.shape[-1]
        inverse_depth = self.predict_inverse_depth(x)

        if f_px is not None:
            f_px = torch.as_tensor(f_px, dtype=torch.float32, device=x.device).squeeze()
        elif self.fov is not None:
            f_px = self.fov.focal_length_px(x)

        depth = None if f_px is None else f_px / (width * inverse_depth)

        if is_image:
            inverse_depth = inverse_depth[0]
            depth = None if depth is None else depth[0]
        return {"inverse_depth": inverse_depth, "depth": depth, "focallength_px": f_px}

    def stream(
        self,
        window: int = DEFAULT_WINDOW,
        overlap: int = DEFAULT_OVERLAP,
        f_px: float | torch.Tensor | None = None,
        *,
        mode: VideoMode = "buffered",
    ) -> DepthStream:
        """Open a session over one video: ``step`` takes each frame and ``flush`` ends the video.

        Args:
            window: Frames attended together: the clip length (buffered) or the cache size,
                the current frame included (unbuffered).
            overlap: Frames shared by consecutive clips, fewer than ``window``; ``buffered`` only.
            f_px: Focal length in pixels of the frames. Defaults to the median estimate over the
                frames so far, kept once ``window`` frames are in.
            mode: ``buffered`` predicts overlapping clips, so a frame also sees later frames and is
                returned once its clip is done; ``unbuffered`` predicts each frame as it arrives,
                from itself and the frames before it.

        Returns:
            A :class:`~depth_pro_2.streaming.DepthStream`, whose ``step`` and ``flush`` return
            lists of the predictions finished, as :meth:`infer_video` yields them.
        """
        return DepthStream(self, window, overlap, f_px, mode=mode)

    def infer_video(
        self,
        frames: Iterable[torch.Tensor],
        window: int = DEFAULT_WINDOW,
        overlap: int = DEFAULT_OVERLAP,
        f_px: float | torch.Tensor | None = None,
        *,
        mode: VideoMode = "buffered",
    ) -> Iterator[Prediction]:
        """Infer depth for a video, one prediction per frame, in order.

        Args:
            frames: Normalized frames ``(3, H, W)``, read lazily.
            window: Frames attended together: the clip length (buffered) or the cache size,
                the current frame included (unbuffered).
            overlap: Frames shared by consecutive clips, fewer than ``window``; ``buffered`` only.
            f_px: Focal length in pixels of the frames. Defaults to the median estimate over the
                first window.
            mode: ``buffered`` processes overlapping clips, so a frame also sees later frames;
                ``unbuffered`` predicts each frame from itself and the frames before it.

        Yields:
            ``inverse_depth`` and ``depth`` in meters, both ``(H, W)``, and ``focallength_px``.
        """
        if f_px is None and self.fov is not None:
            # Unlike a live feed, a file can look ahead: every frame gets the first window's focal.
            frames = iter(frames)
            head = list(itertools.islice(frames, window))
            if not head:
                return
            f_px = self.fov.focal_length_px(torch.stack(head))
            frames = itertools.chain(head, frames)
        stream = self.stream(window, overlap, f_px, mode=mode)
        for frame in frames:
            yield from stream.step(frame)
        yield from stream.flush()
