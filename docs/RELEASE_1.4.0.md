# 1.4.0 — Explicit data preparation and conversion windows

- Reconstruction accepts raw projections, Flat/Dark corrected transmission, and already logarithmic attenuation. The state is recorded in each run. Metadata selects the state; unknown files without calibration require a user selection.
- Calibration and minus-log are skipped according to the actual input state. Processed input has no additional stripe/dezinger/bright-ratio/flat-linear processing in this release. The algorithm list follows validated preparation capabilities, with a second backend check.
- Images → H5 and H5 → Images open dedicated conversion dialogs. Projection H5 files contain actual theta with explicit units. Reconstructed volumes do not acquire fake projection angles.
- TIFF conversion preserves values/dtype. PNG uses a shared stack range. Range/stride export also samples theta consistently.
- Background conversion reports real progress and removes incomplete output after cancellation; existing outputs and sources are protected.
- Fixed Windows log encoding so progress characters cannot turn a completed numerical reconstruction into a reported failure.
- Intelligent analysis remains limited to raw projections. This release does not extend processed input to the SAM3 search/segmentation workflow.

Read [Data preparation guide](DATA_PREPARATION_1.4.0.md). The complete local validation report and per-case logs are stored with the validation data; tests use real Tomo and Lamino subsets rather than a full-volume speed or image-quality benchmark.
