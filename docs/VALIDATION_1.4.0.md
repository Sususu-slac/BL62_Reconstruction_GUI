# Data preparation validation — 2026-11-18

Release 1.4.0. Windows / RTX 6000 Ada GPU0. Real measured Tomo and Lamino subsets, with all measured projection angles retained. This is functional and within-method numerical consistency validation, not full-volume accuracy or speed certification.

Tomo: NMC811_Rod, 901 projections, 8 central rows, horizontal binning 8, original center 1028.5. Lamino: wing3, 1441 projections, 512 central rows then horizontal/vertical binning 8, original center 1071.0, tilt 30°. Three matched input files: raw with calibration, normalized without calibration, attenuation without calibration. Parzen; no stripe/dezinger; iterative methods use 20 iterations. Each full comparison writes four 256×256 slices.

| Method | Raw | Normalized | Attenuation | Max relative L2 |
|---|---|---|---|---|
| fourierrec | pass | pass | pass | 3.46e-07 |
| lprec | pass | pass | pass | 1.17e-07 |
| linerec | pass | pass | pass | 1.84e-07 |
| fbp_cuda | pass | pass | pass | 9.3e-08 |
| sirt_cuda | pass | pass | pass | 6.25e-08 |
| cgls_cuda | pass | pass | pass | 1.33e-05 |
| lamino_linerec | pass | pass | pass | 1.21e-07 |
| lamino_fourierrec | pass | pass | pass | 1.89e-07 |
| lamino_tofu | pass | pass | pass | 1.15e-07 |

All 27 comparisons passed the 0.001 relative-L2 threshold with finite, nonempty outputs. LpRec used its explicit uniform-angle approximation for all states.

Additional checks: 9 center trials, 6 tilt trials, 9 further-binning full runs (128×128), 2 degree/radian comparisons; all passed. Processed center/angle outputs also matched the corresponding raw baseline. A 901-frame TIFF/H5 round trip was reconstructed with FourierRec and matched its original baseline (relative L2 3.37e-7). Eight- and sixteen-bit PNG imports preserved dtype.

115 regression tests passed. Frozen executable self-test covered both conversion dialogs, TIFF numerical round trip and PNG writing. Frozen manual Tofu workflow selected a center and reconstructed four output slices from attenuation input; deployed executable and shared source self-tests passed.

Processed input currently disables additional stripe removal, dezinger, bright-ratio and flat-linear corrections. Intelligent analysis remains raw-only. Binning attenuation is not mathematically equivalent to binning raw intensity before the logarithm. Existing core tests cover raw preprocessing; the new numerical comparison intentionally disables optional filters.

Images → H5 uses Start + Step or CSV/TXT angle input.
