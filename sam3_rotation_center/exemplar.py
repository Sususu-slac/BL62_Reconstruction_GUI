"""Reference-image masks shared by the GUI and the isolated SAM3 worker.

No torch/Qt imports: saving an example never loads a model or modifies its source.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@dataclass
class InstanceExample:
    image: Image.Image
    mask: np.ndarray
    source: str = ''

    def validate(self):
        mask = np.asarray(self.mask)
        if mask.ndim != 2 or mask.shape != (self.image.height, self.image.width):
            raise ValueError('The example mask must match the reference image size')
        if not mask.any():
            raise ValueError('Paint a feature before confirming the example')
        return self


def read_reference(path: str | Path) -> Image.Image:
    """Keep ordinary RGB/8-bit images; normalize scientific gray images once."""
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        array = np.asarray(image)
        if array.dtype == np.uint8:
            if image.mode in ('RGBA', 'LA'):
                background = Image.new('RGBA', image.size, (127, 127, 127, 255))
                return Image.alpha_composite(background, image.convert('RGBA')).convert('RGB')
            return image.convert('RGB').copy()
        if array.ndim != 2:
            raise ValueError('Use an RGB/8-bit image or a single 16/32-bit grayscale image')
        finite = np.isfinite(array)
        if not finite.any():
            raise ValueError('The reference image has no finite pixels')
        low, high = np.percentile(array[finite], (1, 99.9))
        if high <= low:
            raise ValueError('The reference image has no usable intensity range')
        gray = np.zeros(array.shape, np.uint8)
        gray[finite] = np.round(np.clip((array[finite].astype(np.float64)-low)/(high-low), 0, 1)*255).astype(np.uint8)
        return Image.fromarray(gray).convert('RGB')


def mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError('Paint a feature before confirming the example')
    return int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1


def example_preview(example: InstanceExample) -> Image.Image:
    example.validate()
    mask = np.asarray(example.mask, dtype=bool)
    pixels = np.asarray(example.image.convert('RGB')).copy()
    padded = np.pad(mask, 1)
    interior = mask & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    # Inner and outer contours (including erased holes); preserve the image itself.
    pixels[mask & ~interior] = (99, 244, 181)
    x0, y0, x1, y1 = mask_box(mask)
    margin = max(6, int(max(x1-x0, y1-y0)*.18))
    return Image.fromarray(pixels).crop((max(0, x0-margin), max(0, y0-margin),
                                       min(example.image.width, x1+margin), min(example.image.height, y1+margin)))


def isolated_reference(example: InstanceExample) -> tuple[Image.Image, tuple[int, int, int]]:
    """The actual mask affects the input even when native mask encoding is absent.

    Keep source dimensions/feature scale. All unselected pixels, including holes,
    are replaced by a neutral local background BEFORE backbone extraction. Native
    visual-box encoding then reads the selected feature. This is an experimental
    mask-isolation adapter, not a claim of upstream native mask-prompt support.
    """
    example.validate()
    mask = np.asarray(example.mask, dtype=bool)
    pixels = np.asarray(example.image.convert('RGB'))
    x0, y0, x1, y1 = mask_box(mask)
    pad = max(8, int(max(x1-x0, y1-y0)*.25))
    left, top = max(0, x0-pad), max(0, y0-pad)
    right, bottom = min(example.image.width, x1+pad), min(example.image.height, y1+pad)
    # Use only the ring OUTSIDE the bounding box. Erasing an interior feature
    # should change its pixels without changing the neutral background color.
    ring = np.ones((bottom-top, right-left), bool)
    ring[y0-top:y1-top, x0-left:x1-left] = False
    background = pixels[top:bottom, left:right][ring]
    fill = tuple(int(v) for v in np.median(background, axis=0)) if len(background) else (127, 127, 127)
    isolated = np.empty_like(pixels)
    isolated[:] = fill
    isolated[mask] = pixels[mask]
    return Image.fromarray(isolated), fill


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_example(example: InstanceExample, folder: Path) -> dict:
    example.validate()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    image_path, mask_path = folder/'reference.png', folder/'mask.png'
    example.image.convert('RGB').save(image_path)
    Image.fromarray(np.asarray(example.mask, dtype=np.uint8)*255).save(mask_path)
    example_preview(example).save(folder/'preview.png')
    isolated, fill = isolated_reference(example)
    isolated.save(folder/'isolated_reference.png')
    metadata = dict(source=example.source, size=list(example.image.size),
                    mask_pixels=int(np.count_nonzero(example.mask)), box_xyxy=list(mask_box(example.mask)),
                    encoding='mask_isolated_reference_box_features', background_rgb=list(fill),
                    image_sha256=_sha(image_path), mask_sha256=_sha(mask_path))
    metadata_path = folder/'example.json'
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return dict(image=str(image_path.resolve()), mask=str(mask_path.resolve()), metadata=str(metadata_path.resolve()))


def load_example(paths: dict) -> InstanceExample:
    image_path, mask_path = Path(paths['image']), Path(paths['mask'])
    metadata = json.loads(Path(paths['metadata']).read_text(encoding='utf-8'))
    if _sha(image_path) != metadata['image_sha256'] or _sha(mask_path) != metadata['mask_sha256']:
        raise ValueError('The saved reference image or mask has changed; create a new analysis')
    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        return InstanceExample(image.convert('RGB').copy(), np.asarray(mask.convert('L')) > 0,
                               metadata.get('source', '')).validate()
