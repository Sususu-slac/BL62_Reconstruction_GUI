from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import time
from PIL import Image


@dataclass
class Prediction:
    masks: np.ndarray
    scores: np.ndarray


class Sam3ParticleSegmenter:
    def __init__(
        self,
        checkpoint: Path,
        *,
        prompt: str = "particle",
        confidence_threshold: float = 0.2,
        device: str = "cuda",
        mask_chunk_size: int = 8,
        cache_limit_gib: float = 12.0,
    ) -> None:
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as error:
            raise RuntimeError(
                "SAM3 is not installed. Install the Meta SAM3 repository first."
            ) from error

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

        self.prompt = prompt
        self.device = device
        self.mask_chunk_size = max(1, int(mask_chunk_size))
        self.cache_limit_bytes = int(cache_limit_gib * 2**30)
        self.last_telemetry = {}
        self._text_features = None
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.startswith("cuda")
            else nullcontext()
        )
        with context:
            model = build_sam3_image_model(
                checkpoint_path=str(checkpoint),
                load_from_HF=False,
                device=device,
                eval_mode=True,
            )
        # The upstream builder only moves to GPU for the exact string 'cuda'.
        model = model.to(device).eval().requires_grad_(False)
        self.processor = Sam3Processor(
            model,
            device=device,
            confidence_threshold=confidence_threshold,
        )

    def predict(self, image: Image.Image) -> Prediction:
        started = time.perf_counter()
        if self.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(self.device)
        prediction = self._predict_bounded(image)
        # All image-specific GPU tensors are out of scope before trimming cache.
        if self.device.startswith('cuda'):
            if torch.cuda.memory_reserved(self.device) > self.cache_limit_bytes:
                torch.cuda.empty_cache()
            self.last_telemetry = self.memory_snapshot()
        self.last_telemetry['inference_seconds'] = time.perf_counter()-started
        return prediction

    def memory_snapshot(self):
        if not self.device.startswith('cuda'):
            return {}
        return {
            'cuda_allocated_gib': torch.cuda.memory_allocated(self.device)/2**30,
            'cuda_reserved_gib': torch.cuda.memory_reserved(self.device)/2**30,
            'cuda_peak_allocated_gib': torch.cuda.max_memory_allocated(self.device)/2**30,
            'cuda_peak_reserved_gib': torch.cuda.max_memory_reserved(self.device)/2**30,
        }

    @torch.inference_mode()
    def _predict_bounded(self, image: Image.Image) -> Prediction:
        from sam3.model.data_misc import interpolate
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.startswith("cuda")
            else nullcontext()
        )
        with context:
            state = self.processor.set_image(image)
            if self._text_features is None:
                self._text_features = self.processor.model.backbone.forward_text([self.prompt], device=self.device)
            state['backbone_out'].update(self._text_features)
            outputs = self.processor.model.forward_grounding(
                backbone_out=state['backbone_out'], find_input=self.processor.find_stage,
                geometric_prompt=self.processor.model._get_dummy_prompt(), find_target=None)
            probabilities = (outputs['pred_logits'].sigmoid()
                             * outputs['presence_logit_dec'].sigmoid().unsqueeze(1)).squeeze(-1)
            keep = probabilities > self.processor.confidence_threshold
            scores = probabilities[keep].float().cpu().numpy()
            logits = outputs['pred_masks'][keep]
            height, width = image.height, image.width
            masks = np.empty((len(scores), height, width), dtype=bool)
            # Preserve full-resolution bilinear+sigmoid semantics, but never
            # materialize every full-resolution float probability map on GPU.
            for first in range(0, len(scores), self.mask_chunk_size):
                last = min(first+self.mask_chunk_size, len(scores))
                expanded = interpolate(logits[first:last].unsqueeze(1), (height, width),
                                       mode='bilinear', align_corners=False).sigmoid()
                masks[first:last] = (expanded > 0.5).squeeze(1).cpu().numpy()
                del expanded
        return Prediction(masks=masks, scores=scores)
