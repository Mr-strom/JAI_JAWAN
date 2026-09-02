"""
sahi_inference.py — SAHI-style sliced inference for small/distant target detection.

Self-contained: does NOT import from the SAHI package.

Algorithm source:
    repos/03_small_object/sahi/sahi/slicing.py  ── get_slice_bboxes()
    Adapted with attribution below. Original authors: SAHI contributors (MIT License).
    https://github.com/obss/sahi

Why not import SAHI directly:
    The full SAHI package depends on PIL, shapely, tqdm, COCO utils — none of which
    are needed for inference-only use. Extracting the ~30-line slice math keeps the
    production dependency surface minimal and avoids version conflicts.

Pipeline this module provides:
    Frame
      ↓
    get_slice_bboxes()      ← adapted from SAHI slicing.py (pure math, zero deps)
      ↓
    YOLO.predict() per tile ← same model already initialised in TrajectoryEngine
      ↓
    remap boxes to full-frame pixel coordinates
      ↓
    cross-tile NMS merge    ← torchvision.ops.nms (torchvision is ultralytics dep)
      ↓
    _SahiDetections         ← lightweight container implementing the interface that
                               BYTETracker.update() requires (conf, xywh, cls, indexing)
      ↓
    BYTETracker.update()    ← same tracker already in _CameraEngineState, fed the
                               merged SAHI detections instead of model.track() output

All config knobs live in config.py under the SAHI section.
No values are hardcoded here.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ─── Slice bbox generation ────────────────────────────────────────────────────
# Adapted from:  repos/03_small_object/sahi/sahi/slicing.py
# Function:      get_slice_bboxes()
# License:       MIT  (original SAHI repository)
# Changes:       Removed PIL/coco/tqdm imports; kept pure numpy math; added type hints.

def get_slice_bboxes(
    image_height: int,
    image_width: int,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
) -> List[List[int]]:
    """Return tile bounding boxes for slicing a frame into overlapping crops.

    Each entry is [x_min, y_min, x_max, y_max] in pixel coordinates.
    Tiles overlap by (overlap_ratio * slice_size) pixels so targets near tile
    edges are not missed.

    Source: SAHI slicing.py::get_slice_bboxes() — MIT License
    """
    if overlap_height_ratio >= 1.0 or overlap_width_ratio >= 1.0:
        raise ValueError("Overlap ratios must be < 1.0")

    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)

    slice_bboxes: List[List[int]] = []
    y_min = y_max = 0

    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            # Last tile: clamp to boundary, shift back so tile is always full-sized
            if y_max > image_height or x_max > image_width:
                x_max = min(image_width, x_max)
                y_max = min(image_height, y_max)
                x_min = max(0, x_max - slice_width)
                y_min_clamped = max(0, y_max - slice_height)
                slice_bboxes.append([x_min, y_min_clamped, x_max, y_max])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap

    return slice_bboxes


# ─── NMS helper ──────────────────────────────────────────────────────────────

def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """NMS via torchvision (ultralytics transitive dep), numpy fallback if missing."""
    try:
        from torchvision.ops import nms as tv_nms
        return tv_nms(boxes.float(), scores.float(), iou_threshold)
    except ImportError:
        logger.warning("[SAHI] torchvision unavailable — using numpy NMS fallback")
        return _numpy_nms(boxes, scores, iou_threshold)


def _numpy_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Greedy NMS fallback — O(N²), only used when torchvision is absent."""
    b = boxes.cpu().numpy()
    s = scores.cpu().numpy()
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = s.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return torch.tensor(keep, dtype=torch.long)


# ─── ByteTrack-compatible detection container ─────────────────────────────────

class _SahiDetections:
    """Minimal container holding merged SAHI detections.

    Implements exactly the attributes and indexing that BYTETracker._split_detections()
    and BYTETracker.init_track() read:

        results.conf  → np.ndarray  [N]     detection confidences
        results.xywh  → np.ndarray  [N, 4]  cx, cy, w, h in full-frame pixels
        results.cls   → np.ndarray  [N]     class IDs
        results[mask] → _SahiDetections     boolean-index subset
        len(results)  → int

    xyxy is also stored for use in the trajectory engine box loop.
    """

    def __init__(
        self,
        xyxy: np.ndarray,   # [N, 4]  x1,y1,x2,y2 full-frame pixels
        conf: np.ndarray,   # [N]     confidences
        cls: np.ndarray,    # [N]     class ids
    ) -> None:
        self._xyxy = xyxy
        self.conf  = conf
        self.cls   = cls

        # Derive xywh (cx, cy, w, h) — ByteTracker uses this
        cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
        cy = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
        bw = xyxy[:, 2] - xyxy[:, 0]
        bh = xyxy[:, 3] - xyxy[:, 1]
        self.xywh = np.stack([cx, cy, bw, bh], axis=1)

    # ByteTracker calls len(results) and results[bool_mask]
    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask) -> "_SahiDetections":
        return _SahiDetections(self._xyxy[mask], self.conf[mask], self.cls[mask])

    @property
    def xyxy_px(self) -> np.ndarray:
        """Full-frame [x1,y1,x2,y2] in pixels — used by trajectory_engine box loop."""
        return self._xyxy


# ─── Main inference function ──────────────────────────────────────────────────

def run_sahi_inference(
    model,
    frame: np.ndarray,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
    conf_threshold: float,
    iou_threshold: float,
    classes: Optional[List[int]],
    nms_iou_threshold: float,
) -> Optional["_SahiDetections"]:
    """Run tiled YOLO inference and return ByteTrack-compatible merged detections.

    Returns None if no detections found across all tiles.
    The caller (trajectory_engine.py) passes the returned _SahiDetections directly to
    BYTETracker.update() — no other changes to the engine loop are needed.

    Args:
        model:                  Ultralytics YOLO model (same instance as TrajectoryEngine).
        frame:                  Full-resolution BGR frame (numpy).
        slice_height/width:     Tile dimensions in pixels.
        overlap_*_ratio:        Fractional overlap between adjacent tiles.
        conf_threshold:         Per-tile YOLO confidence (lower catches more small targets).
        iou_threshold:          Per-tile YOLO NMS IoU threshold.
        classes:                COCO class IDs to keep (None = all).
        nms_iou_threshold:      IoU threshold for cross-tile duplicate removal.
    """
    h, w = frame.shape[:2]

    # 1. Generate tile coordinates
    tiles = get_slice_bboxes(
        image_height=h,
        image_width=w,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )
    logger.debug("[SAHI] %dx%d frame → %d tiles (%dx%d, ovlp %.1f/%.1f)",
                 w, h, len(tiles), slice_width, slice_height,
                 overlap_width_ratio, overlap_height_ratio)

    # 2. Run YOLO predict (not track) on each tile — no tracker state per tile
    all_xyxy: List[np.ndarray] = []
    all_conf: List[np.ndarray] = []
    all_cls:  List[np.ndarray] = []

    for x_min, y_min, x_max, y_max in tiles:
        tile = frame[y_min:y_max, x_min:x_max]
        if tile.size == 0:
            continue

        results = model.predict(
            tile,
            conf=conf_threshold,
            iou=iou_threshold,
            classes=classes,
            verbose=False,
        )

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            continue

        boxes = results[0].boxes
        xyxy_tile = boxes.xyxy.cpu().numpy().copy()  # [N,4] tile-local pixels

        # Remap from tile-local → full-frame coords
        xyxy_tile[:, 0] += x_min
        xyxy_tile[:, 1] += y_min
        xyxy_tile[:, 2] += x_min
        xyxy_tile[:, 3] += y_min

        all_xyxy.append(xyxy_tile)
        all_conf.append(boxes.conf.cpu().numpy())
        all_cls.append(boxes.cls.cpu().numpy())

    if not all_xyxy:
        return None

    # 3. Merge and cross-tile NMS
    xyxy_all = np.concatenate(all_xyxy, axis=0)   # [M, 4]
    conf_all  = np.concatenate(all_conf, axis=0)   # [M]
    cls_all   = np.concatenate(all_cls,  axis=0)   # [M]

    # Clamp to frame boundaries (floating point/tile edge rounding)
    xyxy_all[:, 0] = np.clip(xyxy_all[:, 0], 0, w)
    xyxy_all[:, 1] = np.clip(xyxy_all[:, 1], 0, h)
    xyxy_all[:, 2] = np.clip(xyxy_all[:, 2], 0, w)
    xyxy_all[:, 3] = np.clip(xyxy_all[:, 3], 0, h)

    # Per-class NMS (offset boxes by class so classes don't suppress each other)
    xyxy_t  = torch.from_numpy(xyxy_all).float()
    conf_t  = torch.from_numpy(conf_all).float()
    cls_t   = torch.from_numpy(cls_all).float()
    offsets = cls_t.unsqueeze(1) * float(w + h)
    keep    = _nms(xyxy_t + offsets, conf_t, nms_iou_threshold)

    kept_idx = keep.cpu().numpy()
    xyxy_kept = xyxy_all[kept_idx]
    conf_kept  = conf_all[kept_idx]
    cls_kept   = cls_all[kept_idx]

    logger.debug("[SAHI] NMS: %d/%d kept", len(kept_idx), len(conf_all))

    return _SahiDetections(xyxy_kept, conf_kept, cls_kept)
