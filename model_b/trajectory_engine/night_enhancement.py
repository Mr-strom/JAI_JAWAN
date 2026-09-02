"""
night_enhancement.py - RetinexFormer-based low-light enhancement.

Extracted from: repos/04_night_vision/RetinexFormer
Paper: Retinexformer: One-stage Retinex-based Transformer for Low-light
       Image Enhancement (ICCV 2023)
Authors: Yuanhao Cai et al.
Source:  https://github.com/caiyuanhao1998/Retinexformer  (MIT License)

What this module does:
    Takes a BGR uint8 numpy frame (from the camera/video), applies RetinexFormer
    low-light enhancement, and returns an enhanced BGR uint8 frame ready for YOLO.

Integration:
    Called from TrajectoryEngine.process() BEFORE YOLO/SAHI detection, only when
    cfg.ENABLE_NIGHT_ENHANCEMENT is True.

Pipeline position:
    Raw frame (BGR, uint8)
        |
    NightEnhancer.enhance(frame)   <- this module
        |
    YOLO11x + SAHI + ByteTrack
        |
    Trajectory Engine

Pretrained weights (LOL-v1 dataset, recommended for general low-light):
    Download from: https://pan.baidu.com/s/13zNqyKuxvLBiQunIxG_VhQ?pwd=cyh2
    File: LOLv1.pth
    Place at path specified in config.NIGHT_WEIGHTS_PATH
    Weight file contains: { 'params': OrderedDict(...) }
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Window size the RetinexFormer attention blocks require.
# Input H and W must be multiples of this value.
_WINDOW_SIZE = 4


class NightEnhancer:
    """Stateless wrapper around a loaded RetinexFormer model.

    Designed to be created once (in TrajectoryEngine.__init__) and reused
    across all frames. Thread-safe for read -- model weights are frozen.
    """

    def __init__(
        self,
        weights_path: str,
        processing_size: Optional[tuple],   # (W, H) to downscale to; None = full res
        device: str = "cuda",
        # RetinexFormer LOL-v1 config (matches pretrained weights)
        n_feat: int = 40,
        stage: int = 1,
        num_blocks: list = None,
    ) -> None:
        if num_blocks is None:
            num_blocks = [1, 2, 2]

        # Import the arch we extracted -- no basicsr, no lmdb, just torch + einops
        from RetinexFormer_arch import RetinexFormer  # local copy

        self._device = device
        self._processing_size = processing_size  # (W, H) or None

        logger.info(
            "[NightEnhancer] Loading RetinexFormer (n_feat=%d, stage=%d, blocks=%s) "
            "on device=%s", n_feat, stage, num_blocks, device
        )

        model = RetinexFormer(
            in_channels=3,
            out_channels=3,
            n_feat=n_feat,
            stage=stage,
            num_blocks=num_blocks,
        )

        # Load weights
        weights_path = str(weights_path)
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"[NightEnhancer] Weights not found: {weights_path}\n"
                "Download RetinexFormer LOL-v1 weights from:\n"
                "  https://pan.baidu.com/s/13zNqyKuxvLBiQunIxG_VhQ?pwd=cyh2\n"
                "Place the .pth file at the path in config.NIGHT_WEIGHTS_PATH"
            )

        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        # Weight file stores state dict under 'params' key
        state_dict = checkpoint.get("params", checkpoint)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            # Some checkpoints are saved with DataParallel prefix 'module.'
            new_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(new_sd)

        model.to(device)
        model.eval()
        self._model = model
        logger.info("[NightEnhancer] Weights loaded OK from %s", weights_path)

    def enhance(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Enhance a single BGR uint8 frame.

        Pipeline:
            BGR uint8 -> RGB float32 [0,1] -> (optional resize down)
            -> pad to multiple of _WINDOW_SIZE -> GPU tensor
            -> RetinexFormer forward -> clamp [0,1] -> (resize back up)
            -> RGB float32 -> BGR uint8

        The entire GPU tensor path stays on CUDA -- only the final output
        is transferred back to CPU as uint8 numpy.

        Args:
            frame_bgr: BGR uint8 numpy array (H, W, 3).

        Returns:
            Enhanced BGR uint8 numpy array -- same spatial size as input.
        """
        orig_h, orig_w = frame_bgr.shape[:2]

        # Resize down for speed
        if self._processing_size is not None:
            proc_w, proc_h = self._processing_size
            frame_small = cv2.resize(frame_bgr, (proc_w, proc_h),
                                     interpolation=cv2.INTER_LINEAR)
        else:
            frame_small = frame_bgr
            proc_h, proc_w = orig_h, orig_w

        # BGR uint8 -> RGB float32 [0, 1]
        rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Numpy -> CUDA tensor  [1, 3, H, W]
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(
            self._device, non_blocking=True
        )

        # Pad to multiple of _WINDOW_SIZE
        _, _, th, tw = tensor.shape
        pad_h = (_WINDOW_SIZE - th % _WINDOW_SIZE) % _WINDOW_SIZE
        pad_w = (_WINDOW_SIZE - tw % _WINDOW_SIZE) % _WINDOW_SIZE
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

        # RetinexFormer inference
        with torch.no_grad():
            enhanced = self._model(tensor)

        # Crop padding, clamp to [0, 1]
        enhanced = enhanced[:, :, :th, :tw].clamp(0.0, 1.0)

        # CUDA tensor -> CPU numpy -> BGR uint8
        out_rgb = (
            enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        ).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        # Resize back to original resolution if we downscaled
        if self._processing_size is not None:
            out_bgr = cv2.resize(out_bgr, (orig_w, orig_h),
                                 interpolation=cv2.INTER_LINEAR)

        return out_bgr
