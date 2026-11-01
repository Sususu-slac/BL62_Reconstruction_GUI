# Instance feature validation — 29 October 2026

Version: 1.1.0. All scientific checks use physical GPU 0, the installed native
SAM3p1 environment and `D:\Chcekpoints\checkpoint_demo_05142026.pt`.

## Automated regression

67 tests passed in 3.575 seconds in native Windows Python (`tomocupy_astra`).
Coverage includes the existing manual reconstruction controls, adaptive search,
scoring/NMS, export, elapsed/progress reporting, and the new mask editor.

New checks exercise real Qt mouse painting, erased holes, undo/redo, empty-mask
rejection, cancellation, Text/Instance switching, cropped outline preview,
16/32-bit reference normalization, immutable job copies, hash verification and
locking reference controls during analysis. Source reference files are preserved.

A pre-existing shutdown callback issue surfaced during the new GUI tests: an
anonymous GPU-monitor callback could emit from an already deleted QObject.
It now uses a bound QObject method so Qt disconnects it during destruction.

## Actual GUI → worker → GPU workflow

Data: `D:\1_Datasets\2_Tomography\hdf5_collection\NMC662_Rod6.hdf5`.
Reference: previously completed reconstruction slice 1026. Target candidates:
slice 1023. The reference and targets are different images.

Test scope: initial center range 1013–1021, initial step 2, final step 0.25,
and reconstruction/segmentation of rows 1023–1024 only.

| Mode | Result | Rounds | Measured centers | Worker elapsed | Saved slices |
|---|---|---:|---:|---:|---:|
| Text: particle | Passed; center 1017.0 | 4 | 30 | 147.304 s | 2 reconstructed + 2 segmented |
| Instance | Passed; center 1016.5 | 3 | 32 | 136.818 s | 2 reconstructed + 2 segmented |
| Instance Safe Stop | Passed | — | — | 12.920 s to GUI completion | No false completed report |

The fixed reference mask contains 696 pixels; reference tokens have shape
`[2, 1, 256]`. Preparing the reference after model loading took 0.880 seconds.
The same model/reference is reused for candidate scoring and final segmentation.

Both workflows produced finite, nonconstant reconstruction TIFFs, nonempty
segmentation labels of matching size, score curves, best images and final
summaries. Original HDF5, reference TIFF and original example-mask files retained
their sizes and modification timestamps.

GUI heartbeat: p99 119.65 ms; maximum 743.95 ms, including startup, saving example
assets and screenshot capture. This is not a claim that every UI operation has
zero latency. Timing runs were not controlled model-speed benchmarks.

This is a bounded integration test, not a full global center search or proof of
physical center accuracy. The 0.5-pixel difference between modes is recorded, not
treated as one method being more accurate. Transfer to unrelated materials and
features remains unvalidated.

Artifacts: `test_outputs\instance_integration_1793317072256197600`.

## Real GPU mask-conditioning check

The same reference was tested with its original painted region and with an
interior erasure of 113 pixels. Both masks have exactly the same bounding box
`[753, 1223, 782, 1253]`, source dimensions and neutral background color.

Two real production adapter instances (no model/inference mocks) produced
different cached visual features: maximum absolute token difference 0.97816646.
On the same target, detections changed from 85 to 87 and the confidence-weighted
score from 70.05534 to 69.72232. This confirms that internal erasure affects the
model, even when the bounding rectangle stays unchanged. It does not imply an
improvement in segmentation quality.

The installed builder still has no native mask encoder. See
[INSTANCE_PROMPT.md](INSTANCE_PROMPT.md) for the explicit mask-isolation adapter.

Artifacts: `test_outputs\instance_mask_gpu_1793317427977606200`.

## Backup

`test_outputs\before_instance_20261029_163159.zip` preserves changed original
source files, the old desktop executable and the old bundled worker sources.
The existing README/manual edits that predated this work were left intact.

Windows build and deployment checks are recorded separately in the corresponding
`test_outputs\instance_*selftest_110.json` and `instance_build_110.log` artifacts.

The frozen distribution and the updated desktop target both passed their native
startup checks: version 1.1.0, two tabs, six manual methods, collapsed Log, Segment
All selected, reconstruction runtime found, worker code and bundled SAM3 present.
The packaged worker sources match the tested sources. An initial frozen check
found missing Conda Qt/Shiboken DLLs; the build script now includes the build
environment's `Library\bin` during dependency discovery. A clean rebuild passed.

The desktop shortcut retains its existing path and icon, pointing to
`D:\BL62_Reconstruction_GUI\BL62_Reconstruction_GUI.exe`. The previous complete
desktop runtime/executable are retained in
`test_outputs\before_instance_20261029_native`. The old window was verified idle
(reconstruction enabled; safe stop and export cancel disabled) before closing.

Local installer: `release\BL62_Reconstruction_GUI_Setup_1.1.0.exe`, 150,679,249 bytes.
SHA-256: `42605e61bfa5c2f16a98f44b70e4b6d020d277f08e146ef275f1762d0d2be711`.
No GitHub release was created as part of this feature-edit task.
