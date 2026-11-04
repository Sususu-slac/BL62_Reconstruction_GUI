# BL62 Reconstruction 1.2.6 — 2026-11-04

This release consolidates the reconstruction updates made on November 4.

## Controls and input defaults

- Advanced Settings now has a compact right-aligned label followed by a small gear button.
- The shared center/tilt candidate viewer is named Parameter Optimization. Apply Selected Parameter applies the selected rotation center or laminography angle.
- Importing a different or updated HDF5 file refreshes row limits, selects the middle detector row for trials, and initializes the rotation center from detector width. The initial center is a starting value, not a measured answer. Choices remain unchanged when the same input is used again.
- Tomography and laminography continue to use their respective reconstruction backends. Fixed-angle center trials, angle trials and full reconstruction remain separate operations.

## Preprocessing

- Vo-all stripe removal is the default for manual reconstruction, automatic center search and final reconstruction.
- Advanced Settings exposes stripe removal, dezinger size/threshold, reconstruction filter, bright ratio, minus-log and flat interpolation.
- Manual and automatic reconstruction share the preprocessing configuration. ASTRA methods run the corresponding GPU preprocessing in external workers.
- Dezinger remains off by default; reproducing a command with dezinger 5 requires selecting size 5.

## Interface responsiveness

- Scan result directories and check TIFF completeness/rewrites in the background.
- Merge each result batch once and request one preview per batch.
- Decode TIFF previews and compute display contrast in the background. Only the latest pending selection is retained; resetting the viewer invalidates stale loads.
- Show Finalizing results while collecting the final outputs. Report completion after that collection finishes.
- Append process log output in chunks.

## Validation and scope

The user confirmed that tomography and laminography reconstruction are working with the correct flat-field calibration. The earlier failed laminography comparison used an incorrect flat field; it does not establish a defect in the reconstruction algorithm.

The 1.2.5 code passed 90 unit/GUI checks, including both geometry controls, preprocessing forwarding, same-size TIFF rewrites, asynchronous final scans, stable-file detection, browsing and stale-preview handling. The 1.2.6 public source also passed all 90 checks before publication.

A native Windows replay used 400 existing 2048×2048 float32 candidate TIFFs. Old per-file preview/list processing blocked the event loop for 130.315 seconds. The new batch UI update took 0.01834 seconds; its maximum observed heartbeat gap was 34.04 ms. A separate asynchronous final scan collected all 400 candidates in 0.234 seconds, with a maximum heartbeat gap of 31.07 ms. These are interface tests, not reconstruction throughput or universal responsiveness guarantees.

The Windows installer and portable ZIP contain the GUI. GPU drivers, reconstruction/SAM3 environments and model weights remain external. Sharing the program files does not share user settings.

See [preprocessing](PREPROCESSING.md), [input defaults](INPUT_PARAMETERS.md), [laminography commands](LAMINOGRAPHY_COMMANDS.md) and [responsiveness details](UI_RESPONSIVENESS_1.2.5.md).
