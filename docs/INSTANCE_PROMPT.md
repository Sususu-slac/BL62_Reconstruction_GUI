# Instance prompting — 1.1.0 test feature

In **Intelligent Data Analysis (Test)**, change **Analysis Target** from **Text**
to **Instance**. Click **Add Image**, choose a reference image, and paint one
complete representative feature. The reference may be a different image from
the reconstructed candidates.

Use **Brush**, **Eraser**, **Undo**, and **Redo**. Brush size is in original-image
pixels. The mouse wheel zooms; the middle mouse button pans; **Fit** fits the
image. Shortcuts: **B**, **E**, **Ctrl+Z**, **Ctrl+Y** / **Ctrl+Shift+Z**.

**Confirm** closes the editor and shows a crop with the selected feature outlined,
including erased holes. An empty mask cannot be confirmed. **Cancel** leaves the
previous example intact. Click **Add Image** again to replace the example.

**Segment All**, **Settings**, **Start Analysis**, and **Safe Stop** keep their
existing behavior. One fixed example is prepared once per analysis and reused
for all center candidates and, when enabled, all reconstructed slices. Text mode
continues to use the text field. The reference object is never added to a target
image or counted in target scores.

## What the installed model actually reads

The installed SAM3 geometry builder has no native mask encoder. This release
uses an explicit experimental adapter:

1. Normalize 16/32-bit grayscale reference images for SAM3; preserve the original
   file. Ordinary RGB/8-bit reference images are used directly.
2. Replace every pixel outside the painted mask, including erased holes, with a
   fixed local background color. Keep the original reference dimensions and scale.
3. Extract native positive-box visual features from the mask-isolated reference
   image at the mask's tight bounding box.
4. Cache those reference visual tokens and apply them to each target. Reference
   coordinates are never used as geometric prompts on a target image. The neutral
   internal word `visual` supplies protocol fields; `particle` is not used as the
   target noun in Instance mode.

Thus changing the painted mask changes the model's actual reference input, even
when its bounding box is unchanged. This does not claim upstream native freehand
mask encoding. Generalization to other materials/features still needs validation.

The confidence-weighted scoring rule, mask NMS, cumulative global Gaussian
filtering, and coarse-to-fine center search are unchanged. A peak is evidence
from the segmentation model, not a physical calibration of the center.

## Saved reference and reproducibility

Each job saves an independent `instance_reference` directory containing:

- `reference.png`: the RGB image used for reference preparation.
- `mask.png`: the full-resolution binary mask, including holes.
- `preview.png`: the cropped outline preview.
- `isolated_reference.png`: the actual mask-isolated image sent to the backbone.
- `example.json`: source path, dimensions, box, mask area, background and hashes.
- `encoding.json`: actual visual token shape, preparation time and encoder mode.

`analysis_job.json` records the selected mode and paths. Hashes detect modified
reference assets; resume fingerprints include the saved example. Editing the
original source file after starting a job does not change that job's saved sample.
The GUI disables reference/mode changes during an active analysis.

Choose the scientific GPU in **Settings**. No additional SAM3 checkpoint or model
installation is needed; the existing shared checkpoint remains in use.
