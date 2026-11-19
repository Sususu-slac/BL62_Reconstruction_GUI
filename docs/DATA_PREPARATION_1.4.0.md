# Data preparation and image/HDF5 conversion — 1.4.0

The Reconstruction tab has **Data Preparation** below the input path:

| Input | Required fields | Reconstruction processing |
|---|---|---|
| Raw projections | projections, theta, Flat, Dark | Existing calibration and selected preprocessing |
| Flat/Dark corrected | projections, theta | Skip calibration; apply minus-log once |
| −Log applied | projections, theta | Skip calibration and minus-log |

Preparation metadata (`data_preparation`, on `/exchange/data` or the file root) selects the state automatically. For older files with neither calibration fields nor preparation metadata, choose the state explicitly. Gray values do not reliably identify the state. Incorrect state selection can produce incorrect reconstructions.

For processed input, additional stripe removal, dezinger, bright-ratio correction and flat-linear correction are disabled in this release. The FBP filter remains selectable. Binning averages the selected input representation: binning already logarithmic values is not the same operation as binning raw intensities then taking the logarithm. Choose the preparation state describing the actual stored pixels. Intelligent Data Analysis still requires raw input with Flat/Dark. No automatic conversion of reconstructed slices into projections is provided.

Only algorithms listed in the installed validated capability table are selectable for processed input. LpRec still requires explicit permission to replace measured angles with its uniform-angle approximation. Tomo and Lamino geometry remain separate. ASTRA trial previews use FourierRec; TomoCuPy Lamino trial previews use LineRec. Tofu retains its own trial backend.

## Images → H5

Open **Convert Data → Images → H5**. Select the image folder; names are sorted naturally (1, 2, …, 10). All images must be single-channel, 2D, with matching shape and dtype. TIFF numerical values are preserved; color images and multiframe files are rejected.

For **Projections**, select preparation state and provide one angle per image, either using start + step or importing a single-column CSV/TXT. Select degrees or radians explicitly. Raw projections require separate Flat and Dark image folders. The file stores `/exchange/data`, `/exchange/theta` with units, preparation metadata, and actual calibration fields if raw. Projection acquisition angles must describe the actual scan.

For **Reconstructed slices**, the volume is stored at `/reconstruction/data`. It has no projection theta and cannot be used as reconstruction input.

Choose a new output H5 name. Source files are not modified. The conversion runs in the background; cancellation removes its incomplete output. Existing outputs are not overwritten.

## H5 → Images

Select an H5 file, choose a numerical 2D/3D dataset, and set start (included), end (excluded), and stride. Select a **new** output folder.

- TIFF preserves the original numerical dtype and values.
- PNG maps the finite minimum/maximum of all selected frames into one shared 8-bit or 16-bit display range. This changes numerical values. The range is recorded in `conversion.json`.
- Exporting `/exchange/data` also writes the correspondingly sampled `theta.csv` with a comment specifying the angle units.
- Progress reports range measurement and writing separately. Cancel/close requests stop between frames and remove incomplete outputs. Closing the main window waits for conversion workers to stop safely.

## Implementation note

The installed TomoCuPy reader always allocates Flat/Dark buffers. For processed input, an output-local external-link adapter supplies neutral compatibility buffers, explicitly marked as **not measured calibration**. A process-local entry bypasses the calibration function completely. The source H5 is read-only; installed TomoCuPy code is unchanged. Adapter files link to their input and are not standalone archival data.

See the accompanying validation report for the actual test matrix, data subsets, timings, and limits. Passing subset tests does not guarantee physical correctness for every acquisition geometry or arbitrary preprocessing pipeline.
