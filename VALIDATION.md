# Release 1.0.0 validation

Validated on Windows with two NVIDIA RTX 6000 Ada 48 GB GPUs. Raw input: `NMC811_Rod.hdf5` (901 × 2048 × 2048, uint16). Original HDF5 size/mtime and checkpoint size/mtime remain unchanged.

## Automated and GUI tests

- 46 automated tests passed, including cumulative global filtering, NMS, direct search buttons, mode-dependent parameters, per-run outputs, and a concurrent Windows progress-file reader/writer regression test.
- Five real input views passed: Projection, Flat, Dark, Corrected and Attenuation.
- Preview binning 1×1, 2×2, 4×4 and 8×8 passed.
- Real asynchronous GUI exports passed: raw uint16 TIFF, corrected float32 TIFF and display uint16 PNG.

| Manual GUI case | Output | Time (s) |
|---|---|---:|
| tomography_full_fourierrec | 5 · [2048, 2048] | 11.76 |
| tomography_full_lprec | 5 · [2048, 2048] | 17.91 |
| tomography_full_linerec | 5 · [2048, 2048] | 13.34 |
| tomography_full_fbp_cuda | 5 · [2048, 2048] | 3.39 |
| tomography_full_sirt_cuda | 5 · [2048, 2048] | 3.72 |
| tomography_full_cgls_cuda | 5 · [2048, 2048] | 4.17 |
| tomography_center_fourierrec | 2 · [2048, 2048] | 11.07 |
| laminography_full_lamino_fourierrec | 1 · [256, 256] | 50.51 |
| laminography_full_lamino_linerec | 1 · [256, 256] | 33.54 |
| laminography_angle_lamino_linerec | 2 · [256, 256] | 29.03 |
| tomography_stop_sirt_cuda | Stopped ·  | 1.62 |

Timings are validation wall times on this workstation, not controlled method benchmarks. Manual tests used GPU 0 while the global automatic test used GPU 1; CPU and disk were shared.

## Automatic GUI workflow

- Fresh full-domain search: center range 0–2047, detector row 1023, initial step 20 px, final step 0.25 px, 6 rounds and 268 unique candidates.
- GUI start → center search → 5 reconstructed slices (rows 1021–1025) → segmentation → completion: 461.21 seconds in the worker, 464.38 seconds observed by the GUI test.
- Recommended center: **1028.50 px**. The earlier user-confirmed reconstruction is 1028.75 px; difference **0.25 px** (one sampling interval). An earlier native run returned 1028.75. The saved summary identifies competing peaks; no claim of physically calibrated 0.25 px accuracy is made.
- Five reconstruction/label TIFFs and packed instance masks audited: **721 instances**. Every recorded instance area and confidence agrees with its saved mask. Overlapping masks are retained in the NPZ files.
- Optional segmentation disabled: separate GUI run passed (5 reconstructions, zero segmentation outputs).
- Manual Safe Stop and automatic startup cancellation passed; cancelled runs have no success summary. Automatic cancellation completed about 10 seconds after request during module startup.
- During the full automatic GUI test, the 50 ms heartbeat had a 283 ms maximum gap and 133 ms p99 gap. The broader manual/automatic test had a 1639 ms maximum gap and 129 ms p99; short pauses remain during original input/result display work.
- Model loading was 23.73 seconds in the fresh automatic test; the same model was reused for all center candidates and full-output segmentation.

## Scope

The automatic workflow was validated end to end on **five output slices**, not an entire 2048-slice volume. Global center search still covered the original detector width. Laminography remains a manual feature; intelligent angle estimation is outside this release. Other workstations and clean environment installation have not been tested on separate hardware.

## Windows release executable

The standalone GUI build uses separate worker source directories so Python 3.12 display extensions cannot shadow the Python 3.10 reconstruction environment. See the final release check below.

- Final packaged EXE: manual FBP_CUDA reconstructed five slices successfully.
- The same EXE then ran a fresh center search using its **bundled SAM3 source**, reconstructed five slices, segmented all five, and displayed the best image and score curve. Recommended center: 1028.50 px.
- Release executable launch and runtime discovery passed; source worker files are under `_internal/worker_code`, separated from display extension modules.
