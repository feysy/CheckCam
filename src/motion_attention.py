"""
Motion-Attention Frame Preprocessing.

Computes a motion mask across sequential keyframes using cv2.absdiff and morphological
dilation, then attenuates static background elements (via Neutral Gray blending or
Gaussian Blur). Ensures Vision Transformers (SigLIP) focus attention on active movement zones
rather than high-contrast static background features (e.g., car headlights sweeping across walls).
"""

import cv2
import numpy as np
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()


@dataclass
class MotionAttentionConfig:
    """Configuration parameters for Motion-Attention preprocessing."""
    enabled: bool = True
    mode: str = "gray"  # "gray" or "blur"
    difference_threshold: int = 25  # Intensity difference threshold (0-255)
    dilation_iterations: int = 3  # Dilation iterations to bridge gaps in silhouette
    dilation_kernel_size: int = 5  # Kernel size for morphological dilation
    gray_blend_ratio: float = 0.8  # Blend ratio for neutral gray (0.0 to 1.0)
    blur_kernel_size: int = 31  # Kernel size for Gaussian blur background (odd int)
    mask_blur_kernel_size: int = 15  # Kernel size for smooth alpha edge blending (odd int)
    min_motion_area_pct: float = 0.05  # Minimum % of frame area required to apply masking


class MotionAttentionPreprocessor:
    """
    Applies motion-attention masking to keyframes before feature extraction.
    Reuses memory buffers across invocations to minimize memory fragmentation on edge devices (NFR-2).
    """

    def __init__(self, config: MotionAttentionConfig | None = None):
        self.config = config or MotionAttentionConfig()
        # Memory buffer cache for numpy arrays: key -> np.ndarray
        self._buffers: dict[str, np.ndarray] = {}

    def _get_buffer(self, key: str, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Reuse existing numpy array buffer if shape and dtype match to prevent allocation thrashing."""
        buf = self._buffers.get(key)
        if buf is None or buf.shape != shape or buf.dtype != dtype:
            buf = np.empty(shape, dtype=dtype)
            self._buffers[key] = buf
        return buf

    def update_config(self, settings: dict):
        """Update runtime configuration dynamically from database settings."""
        if not isinstance(settings, dict):
            return

        ma = settings.get("motion_attention", settings)
        if "enabled" in ma:
            v = ma["enabled"]
            self.config.enabled = v if isinstance(v, bool) else str(v).lower() in ("true", "1", "yes")
        if "mode" in ma:
            m = str(ma["mode"]).strip().lower()
            if m in ("gray", "blur"):
                self.config.mode = m
        if "difference_threshold" in ma:
            self.config.difference_threshold = max(1, min(int(ma["difference_threshold"]), 255))
        if "dilation_iterations" in ma:
            self.config.dilation_iterations = max(1, min(int(ma["dilation_iterations"]), 20))
        if "dilation_kernel_size" in ma:
            k = max(3, min(int(ma["dilation_kernel_size"]), 31))
            self.config.dilation_kernel_size = k if k % 2 == 1 else k + 1
        if "gray_blend_ratio" in ma:
            self.config.gray_blend_ratio = max(0.0, min(float(ma["gray_blend_ratio"]), 1.0))
        if "blur_kernel_size" in ma:
            b = max(3, min(int(ma["blur_kernel_size"]), 101))
            self.config.blur_kernel_size = b if b % 2 == 1 else b + 1
        if "mask_blur_kernel_size" in ma:
            mb = max(3, min(int(ma["mask_blur_kernel_size"]), 51))
            self.config.mask_blur_kernel_size = mb if mb % 2 == 1 else mb + 1
        if "min_motion_area_pct" in ma:
            self.config.min_motion_area_pct = max(0.0, min(float(ma["min_motion_area_pct"]), 100.0))

        logger.info(
            "motion_attention_config_updated",
            enabled=self.config.enabled,
            mode=self.config.mode,
            difference_threshold=self.config.difference_threshold,
            dilation_iterations=self.config.dilation_iterations,
            gray_blend_ratio=self.config.gray_blend_ratio,
            blur_kernel_size=self.config.blur_kernel_size,
        )

    def compute_motion_masks(self, frames: list[np.ndarray]) -> list[np.ndarray | None]:
        """
        Compute dilated motion mask for each keyframe.

        Uses a two-pass approach:
          1. Global accumulated mask: union of all sequential diffs to find
             regions with *any* motion across the clip.
          2. Per-frame mask: intersection of the per-frame diff with the
             accumulated mask, ensuring isolated transient flashes (headlights)
             don't dominate when they only appear between one pair of keyframes.

        The accumulated mask acts as a spatial prior — regions that change in
        multiple pairs are boosted, while single-pair transients are suppressed.
        """
        n = len(frames)
        if n < 2:
            return [None] * n

        h, w = frames[0].shape[:2]
        gray_shape = (h, w)

        # 1. Convert all frames to grayscale using reusable buffers
        gray_frames: list[np.ndarray] = []
        for i, frame in enumerate(frames):
            buf = self._get_buffer(f"gray_{i}", gray_shape, np.uint8)
            cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY, dst=buf)
            gray_frames.append(buf)

        # Structuring element for morphological dilation
        k_size = self.config.dilation_kernel_size
        if k_size % 2 == 0:
            k_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

        diff_buf = self._get_buffer("diff", gray_shape, np.uint8)
        thresh_buf = self._get_buffer("thresh", gray_shape, np.uint8)

        # --- Pass 1: build per-pair binary masks and accumulate ---
        # Accumulator counts how many sequential pairs show motion at each pixel
        accumulator = self._get_buffer("accumulator", gray_shape, np.float32)
        accumulator.fill(0.0)
        pair_masks: list[np.ndarray] = []

        for i in range(n - 1):
            cv2.absdiff(gray_frames[i], gray_frames[i + 1], dst=diff_buf)
            cv2.threshold(
                diff_buf,
                self.config.difference_threshold,
                255,
                cv2.THRESH_BINARY,
                dst=thresh_buf,
            )
            pair_mask = thresh_buf.copy()
            pair_masks.append(pair_mask)
            # Add normalized contribution to accumulator (each pair contributes 1.0)
            accumulator += (pair_mask.astype(np.float32) / 255.0)

        # --- Pass 2: consistency filter ---
        # Regions with motion in fewer than half the pairs are considered transient
        # (e.g., a headlight flash appears in 1 of 2 pairs → suppressed)
        # For n=3 keyframes (2 pairs): threshold = 1.0 → must appear in both
        # For n=4 keyframes (3 pairs): threshold = 1.5 → at least 2 of 3
        num_pairs = n - 1
        if num_pairs >= 2:
            consistency_threshold = num_pairs * 0.5
        else:
            # Only 1 pair — can't filter, keep everything
            consistency_threshold = 0.5

        # Build sustained motion mask: areas with consistent motion across pairs
        sustained_mask = self._get_buffer("sustained", gray_shape, np.uint8)
        sustained_mask[:] = (accumulator >= consistency_threshold).astype(np.uint8) * 255

        # Dilate the sustained mask to fill gaps
        dilated_sustained = self._get_buffer("dilated_sustained", gray_shape, np.uint8)
        cv2.dilate(
            sustained_mask,
            kernel,
            dst=dilated_sustained,
            iterations=self.config.dilation_iterations,
        )

        # --- Pass 3: per-frame masks combined with sustained mask ---
        masks: list[np.ndarray | None] = []
        for i in range(n):
            # Per-frame difference: union of adjacent pair diffs
            if i == 0:
                per_frame = pair_masks[0]
            elif i == n - 1:
                per_frame = pair_masks[-1]
            else:
                per_frame = cv2.max(pair_masks[i - 1], pair_masks[i])

            # Intersect per-frame motion with dilated sustained mask
            # This keeps only the motion regions that are part of sustained movement
            combined = self._get_buffer(f"combined_{i}", gray_shape, np.uint8)
            cv2.bitwise_and(per_frame, dilated_sustained, dst=combined)

            # Final dilation for this frame
            dilated_buf = self._get_buffer(f"dilated_{i}", gray_shape, np.uint8)
            cv2.dilate(
                combined,
                kernel,
                dst=dilated_buf,
                iterations=max(1, self.config.dilation_iterations // 2),
            )

            # Check minimum motion area
            motion_pixels = cv2.countNonZero(dilated_buf)
            motion_pct = (motion_pixels / dilated_buf.size) * 100.0

            if motion_pixels == 0 or motion_pct < self.config.min_motion_area_pct:
                logger.debug(
                    "motion_mask_empty_fallback",
                    frame_index=i,
                    motion_pct=f"{motion_pct:.3f}%",
                    min_pct=self.config.min_motion_area_pct,
                )
                masks.append(None)
            else:
                masks.append(dilated_buf.copy())

        return masks

    def attenuate_background(self, frame: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        """
        Attenuate background regions (neutral gray or gaussian blur) using smooth alpha blending.
        If mask is None, falls back to the unmodified raw frame (FR-3.2).
        """
        if mask is None:
            return frame

        h, w = frame.shape[:2]
        float_shape = (h, w, 3)

        # 1. Smooth the binary mask edges for seamless alpha transition (FR-2.2)
        k_smooth = self.config.mask_blur_kernel_size
        if k_smooth % 2 == 0:
            k_smooth += 1
        smooth_buf = self._get_buffer("mask_smooth", (h, w), np.uint8)
        cv2.GaussianBlur(mask, (k_smooth, k_smooth), 0, dst=smooth_buf)

        # Normalized floating-point alpha channel: (h, w, 1) in range [0.0, 1.0]
        alpha = self._get_buffer("alpha", (h, w, 1), np.float32)
        np.divide(smooth_buf[:, :, np.newaxis], 255.0, out=alpha)

        # 2. Foreground as float32
        fg = self._get_buffer("fg_float", float_shape, np.float32)
        np.copyto(fg, frame)

        # 3. Attenuated background
        bg = self._get_buffer("bg_float", float_shape, np.float32)
        if self.config.mode == "blur":
            k_blur = self.config.blur_kernel_size
            if k_blur % 2 == 0:
                k_blur += 1
            blurred = self._get_buffer("blurred_bg", frame.shape, np.uint8)
            cv2.GaussianBlur(frame, (k_blur, k_blur), 0, dst=blurred)
            np.copyto(bg, blurred)
        else:
            # Neutral Gray Mode: blends middle-gray (128, 128, 128) with configurable percentage
            blend_ratio = self.config.gray_blend_ratio
            gray_val = 128.0
            # bg = frame * (1 - blend_ratio) + 128 * blend_ratio
            if blend_ratio >= 0.999:
                bg.fill(gray_val)
            else:
                np.multiply(fg, 1.0 - blend_ratio, out=bg)
                bg += gray_val * blend_ratio

        # 4. Smooth Alpha Blending: result = fg * alpha + bg * (1.0 - alpha)
        out_float = self._get_buffer("out_float", float_shape, np.float32)
        inv_alpha = self._get_buffer("inv_alpha", (h, w, 1), np.float32)
        np.subtract(1.0, alpha, out=inv_alpha)

        # fg * alpha + bg * inv_alpha
        np.multiply(fg, alpha, out=out_float)
        out_float += bg * inv_alpha

        # Clip and convert back to uint8
        out_uint8 = self._get_buffer("out_uint8", frame.shape, np.uint8)
        np.clip(out_float, 0, 255, out=out_float)
        np.copyto(out_uint8, out_float.astype(np.uint8))

        return out_uint8.copy()

    def process_keyframes(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        Full pipeline: generates motion masks and applies background attenuation across keyframes.
        Falls back to raw frames if disabled or if motion mask is empty.
        """
        if not self.config.enabled or len(frames) < 2:
            return frames

        masks = self.compute_motion_masks(frames)
        focused_frames: list[np.ndarray] = []

        for frame, mask in zip(frames, masks):
            focused = self.attenuate_background(frame, mask)
            focused_frames.append(focused)

        return focused_frames
