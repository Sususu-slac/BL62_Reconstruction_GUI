# Version 1.3.0 — manual Tofu/UFO GPU laminography

Manual laminography can now use Tofu/UFO GPU for center trials, tilt trials and reconstruction. The existing fields use original detector pixels and the same signed tilt convention; output intervals, GPU selection, preprocessing, candidate selection and the result browser are shared. Each execution has isolated input/cache/output metadata. Intelligent Data Analysis is unchanged and was not run as part of this feature's GPU validation.

The optional runtime is a clean, account-local WSL2 image built from Ubuntu Base 24.04.5 and an isolated Python environment, not an export of a personal distribution. It includes the GPU compatibility plugin, its source, licenses and an offline self-check. Its VHD is placed on D: when available. Windows accounts share the archive but register private runtime instances. Installation validates SHA256 before import.

## Validation

- 108 Windows unit/smoke tests passed, including existing analysis regressions, geometry conversions, inclusive search values, wrapped/radian theta, packaging, cancellation isolation and preview teardown and exact selection of three-decimal tilt candidates.
- Synthetic HDF5 inputs exercised positive and negative tilt, different centers, nonuniform angles, radian units, Chinese/spaced input names, both GPUs and distinct outputs. Center and angle candidates were individually different. GPU1 selection was independently checked against the corresponding Windows CUDA PCI identity.
- The shipped frozen GUI completed all three manual workflows, applied selected center/angle values, browsed output and safely stopped after two completed TIFFs. The automatic stage was explicitly skipped.
- A clean runtime import was tested independently of the developer's Ubuntu installation. Packaged external workers completed center/angle trials and reconstruction against that runtime. A native tomography reconstruction also passed as a regression check.
- Complete coarse Wing3 volume: all 1441 measured projections, 8×8 spatial binning, 296×256×256 output, three forced output slabs. Correlation with the previously validated UFO result: 0.999999999999; relative L2 difference 9.547e-07.
- Original-resolution Wing3 check: output rows 1180–1195, two blocks of eight 2048×2048 slices, original center 1070.5 and +30° tilt. Correlation with the accepted TomoCuPy reconstruction: 0.99999043; relative L2 difference 0.00382. The original HDF5 size and modification timestamp were unchanged.
- Original-resolution viewing: 16 frames browsed and close-during-load tested; maximum observed event-loop timer gap 0.063 s. Synthetic workflow timer gap was 0.125 s. These are local offscreen GUI checks, not a compositor/frame-rate benchmark.
- Safe Stop during reading completed in about 1.8 s; stopping after two saved images completed in under one second and retained those complete images. A timed-out cancellation helper cannot hold the native controller open or terminate unrelated processes.

## Timing and scope

The original-resolution development run took approximately 54.0 minutes end to end, including preparation of the entire projection dataset and temporary-data staging. The two UFO execution blocks together took about 77 seconds, including their input/output. These timing boundaries differ from the earlier prepared-cache benchmark. Allocation, reading, preprocessing, filtering, flushing and staging are now recorded separately in task metadata.

This validates a complete coarse volume and 16 original-resolution slices. The complete 2366-slice original-resolution volume was not repeated, and no full-volume speed claim is made. Late changes only add preview teardown handling, timing/progress records and bounded cleanup; original-resolution numerical reconstruction was not rerun for those diagnostic changes.

## Operational limits

Tofu is offered only under manual Laminography. An even effective width ≥32 and height ≥16 are currently required. Measured theta is preserved; no uniform linspace silently replaces it. The compatibility implementation uses explicit zero padding outside the detector and float-buffer bilinear sampling, not unsupported OpenCL image objects. No CPU fallback is silently substituted.

See [setup and usage](TOFU_GPU.md). Detailed local test artifacts remain outside the public source repository; research images, HDF5 inputs, checkpoints and conversation data are not bundled.
