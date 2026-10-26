# BL62 Reconstruction GUI

Windows GUI for manual tomography/laminography reconstruction and SAM3-assisted automatic tomography analysis.

## Windows installation

1. Install the NVIDIA driver and Miniforge.
2. Run `BL62_Reconstruction_GUI_Setup_1.0.0.exe`, or extract the portable ZIP.
3. Use **Install Environments** to prepare two isolated native Windows environments:
   - `tomocupy_astra`: reconstruction (Python 3.10).
   - `SAM3p1`: SAM3 inference (Python 3.12, PyTorch 2.10.0 CUDA 12.8).
4. Launch **BL62 Reconstruction**. WSL is not required by this release.
5. In **Settings → SAM3**, select your checkpoint if the central default is unavailable.

The GUI executable includes Qt and its display dependencies. GPU reconstruction and SAM3 run in the external environments. The installer does not include GPU drivers, Conda or model weights. Creating new GPU environments downloads several GB.

The shared checkpoint defaults to `D:\Chcekpoints\checkpoint_demo_05142026.pt`. The existing repository `D:\SAM3p1\sam3` is preferred; the Windows bundle also includes the tested SAM3 inference source. Model weights and scientific datasets are never uploaded with the application.

## Reconstruction

The original three-column layout and data preview/export tools are retained.

- Choose Tomography or Laminography; supported parameters switch automatically.
- Enter the rotation center, row range, binning and GPU.
- **Find Center** and **Find Angle** run independent manual sweeps. Gear buttons configure their range and step.
- **Run Reconstruction** reconstructs the selected rows into a new run directory.
- A manual center sweep uses the selected TomocuPy method; if an ASTRA method is selected, its sweep uses FourierRec and preserves the selected full-reconstruction method.
- `End row (exclusive)` is not included: rows 0–2048 produce rows 0–2047.
- GPU selection displays device names and memory. Usage/VRAM refresh in the background.
- **Log** is collapsed by default. Each new job clears its displayed log.

Six tomography methods: FourierRec, LpRec, LineRec, FBP_CUDA, SIRT_CUDA and CGLS_CUDA. Laminography uses FourierRec or LineRec; manual angle/center searches use LineRec. LpRec creates an external-link HDF5 view with uniform angles; raw input is retained.

## Intelligent Data Analysis (Test)

Select an HDF5 file, enter an **Analysis Target** such as `particle`, and click **Start Analysis**. **Segment All** is enabled by default. Optional parameters are under **Settings**.

1. Find the rotation center using the middle detector row, cumulative global Gaussian filtering and confidence-weighted SAM3 scores.
2. Reconstruct the requested rows with the recommended center.
3. If enabled, segment every reconstructed output slice.
4. Save reports, reconstructed TIFFs, segmentation label TIFFs and instance statistics.

By default a new `{input_name}_analysis_{timestamp}` directory is created next to the input. Settings allow a custom parent folder, row range, checkpoint, GPU, initial search range and step. This release automates **tomography rotation-center search**, not laminography angle estimation.

The right panel shows the current stage/round progress, the best reconstruction, its exact center, current round/step, and the cumulative score curve. Best-image/curve refresh defaults to 3 seconds. Search progress is for the current round because adaptive refinement changes the total candidate count.

A model stays resident throughout one automatic job, including optional segmentation, and is released when the worker exits. Safe Stop cancels only this job and its child processes. Partial results remain identifiable; a stopped/failed job is never marked completed. A flat or empty score curve is a failed center estimate.

Segmentation saves per-slice integer label TIFFs and `instances.csv`. Packed instance-mask NPZ files preserve overlaps; decode with `np.unpackbits(data['masks_packed'], axis=1)[:, :height*width].reshape(-1,height,width)`. IDs are local to each slice, not 3D particle tracking.

## Source and validation

Start source with `run_gui.bat` in the `tomocupy_astra` environment.

```bat
python -m unittest discover -s tests -p "test_*.py"
python tests\integration_workflows.py --input DATA.hdf5 --output TEST_OUTPUT --gpu 0
python tests\integration_workflows.py --input DATA.hdf5 --output GLOBAL_TEST --gpu 1 --auto-only --global-search --segment-all
```

Build the Windows executable and installer with `build_windows.ps1` (PyInstaller and Inno Setup 6 required). The build script prepares `runtime_sam3` from the pinned upstream commit automatically when needed (`python prepare_sam3_source.py`). Checkpoint binaries remain external.

See `VALIDATION.md` for actual release tests and their scope. See `THIRD_PARTY_NOTICES.md` and `licenses/SAM_LICENSE` for SAM3 attribution and terms.
