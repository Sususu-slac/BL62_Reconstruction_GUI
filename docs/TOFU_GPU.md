# Manual laminography with Tofu/UFO GPU

Release 1.3.0 adds **Tofu/UFO GPU (Experimental)** to the Laminography Algorithm list. Intelligent Data Analysis is unchanged.

Select the input HDF5, output root, GPU, rotation center, tilt and binning as usual. Find Center, Find Angle and Run Reconstruction all use the selected Tofu backend. Centers are entered in original detector pixels; tilts are entered in the same degrees as TomoCuPy. Internal UFO coordinates are converted automatically. Search intervals include the upper endpoint when it falls on the requested step.

Choose a candidate image and use Apply Parameter to transfer its original-pixel center or tilt to reconstruction. The selected value and its source image are saved in JSON alongside that isolated run's results. Input paths and theta are read for every job; a previous experiment's cache is never substituted.

## Runtime installation

Existing Windows TomoCuPy/ASTRA preprocessing is reused. The optional GPU computation runs in an isolated WSL2 distribution named BL62-Tofu. It has no dependence on a developer's personal Ubuntu distribution, home directory, datasets or checkpoints.

Place BL62-Tofu-GPU-1.0.0.tar.gz and its .sha256 file in D:\BL62_Runtime\Tofu. On first use, the GUI worker can run setup_tofu_runtime.ps1 automatically. Setup validates the checksum, imports a private distribution for the current Windows account and runs an offline GPU self-check. The default virtual disk is on D: when available. The runtime settings are saved under the current account's APPDATA\BL62_Reconstruction\tofu_runtime.json. Each Windows account needs its own one-time WSL registration; the archive may be shared. WSL2 and an NVIDIA driver with WSL GPU support are prerequisites.

## Data and progress

Flat/dark correction, dezinger, stripe removal, bright ratio, minus-log and the selected filter follow the existing preprocessing controls. Only one input frame and one bounded sinogram band are materialized at a time. The initial Tofu preprocessing band is 32 rows; GPU output slabs are chosen from available memory. Temporary projection storage is placed inside the isolated output run and the D:-hosted WSL runtime, then removed on completion or cancellation. Completed TIFFs are committed atomically and appear in the existing viewer.

Progress identifies environment checking, input reading/binning, preprocessing, filtering, GPU staging, projection processing and saving. Safe Stop targets only the current job, retains completed output slices and cleans its temporary cache. It does not terminate another reconstruction or the whole WSL distribution.

## Implementation and current limits

The locally validated UFO compatibility plugin replaces unsupported image objects with float buffers and bilinear sampling. It retains UFO geometry and burst accumulation, explicitly pads outside the detector with zero, uses synchronous execution and smaller GPU thread blocks, and requires an actual GPU rather than silently falling back to the CPU. Measured theta, including nonuniform spacing and declared radian units, is passed explicitly; the backprojection uses the existing equal-projection-weight convention.

Current effective images require an even width of at least 32 and a height of at least 16 pixels. This release enables Tofu only for manual laminography. It does not add Tofu to tomography or intelligent analysis.

See the release validation record for the exact tested volumes and original-resolution coverage; a full-mode test on a selected output interval is not a repeat of an entire original-resolution volume.
