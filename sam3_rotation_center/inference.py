from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import time
import types
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


class Sam3InstanceSegmenter(Sam3ParticleSegmenter):
    """One frozen reference mask reused for every candidate and output slice.

    The installed SAM3 geometry encoder has no native mask encoder. Instead,
    isolate painted pixels BEFORE backbone extraction, encode their positive
    visual box on the REFERENCE image, and inject those cached visual tokens.
    Never apply the reference box/mask coordinates to a target image.
    """
    def __init__(self, checkpoint: Path, *, example, **kwargs):
        from .exemplar import isolated_reference, mask_box
        kwargs.pop('prompt', None)
        super().__init__(checkpoint, prompt='visual', **kwargs)
        started = time.perf_counter()
        example.validate()
        image, fill = isolated_reference(example)
        x0, y0, x1, y1 = mask_box(example.mask)
        processor, network = self.processor, self.processor.model
        original_encode = network._encode_prompt
        context = torch.autocast('cuda', dtype=torch.bfloat16) if self.device.startswith('cuda') else nullcontext()
        with torch.inference_mode(), context:
            state = processor.set_image(image)
            self._text_features = network.backbone.forward_text(['visual'], device=self.device)
            state['backbone_out'].update(self._text_features)
            prompt = network._get_dummy_prompt()
            box = [(x0+x1)/2/image.width, (y0+y1)/2/image.height,
                   (x1-x0)/image.width, (y1-y0)/image.height]
            prompt.append_boxes(torch.tensor(box, device=self.device, dtype=torch.float32).reshape(1, 1, 4),
                                torch.ones((1, 1), device=self.device, dtype=torch.bool))
            tokens, attention, _ = original_encode(state['backbone_out'], processor.find_stage,
                                                    prompt, encode_text=False)
            self.reference_tokens = tokens.detach().clone()
            self.reference_attention = attention.detach().clone()
        self.instance_prepare_seconds = time.perf_counter()-started
        self.instance_metadata = dict(encoding='mask_isolated_reference_box_features',
            box_xyxy=[x0,y0,x1,y1], mask_pixels=int(np.count_nonzero(example.mask)), background_rgb=list(fill),
            reference_size=list(example.image.size), token_shape=list(self.reference_tokens.shape),
            native_mask_encoder_available=network.geometry_encoder.mask_encoder is not None,
            prepare_seconds=self.instance_prepare_seconds, target_text='visual', reference_source=example.source)
        reference_tokens, reference_attention = self.reference_tokens, self.reference_attention

        def encode_with_reference(owned_network, backbone_out, find_input, geometric_prompt, **encode_kwargs):
            if geometric_prompt.box_embeddings.shape[0] or geometric_prompt.point_embeddings.shape[0]:
                raise RuntimeError('Reference coordinates must never be applied to a target image')
            encode_kwargs.update(visual_prompt_embed=reference_tokens,
                                 visual_prompt_mask=reference_attention, encode_text=True)
            return original_encode(backbone_out, find_input, geometric_prompt, **encode_kwargs)

        # Scoped to this worker's owned model; installed SAM3 source is untouched.
        network._encode_prompt = types.MethodType(encode_with_reference, network)
