# Laminography command contract

Reviewed against the installed tomocupy_astra reader and recon_steps implementation, 2026-11-04. This describes actual TomocuPy arguments; GUI worker arguments also carry job metadata and use a different interface.

| Purpose | reconstruction-type | Algorithm | Slice selector | Sweep parameters |
|---|---|---|---|---|
| Manual center candidates | try | linerec | nsino | center-search-width, center-search-step |
| Manual tilt-angle candidates | try_lamino | linerec | nsino | lamino-search-width, lamino-search-step |
| Full reconstruction | full | linerec or fourierrec | lamino-start-row, lamino-end-row | None |

All three use recon_steps, a specific input and unique output directory, rotation-axis, binning, lamino-angle, dtype, save-format and the selected preprocessing values. Center candidates hold the lamino angle fixed. Angle candidates hold the rotation center fixed.

Preprocessing: remove-stripe-method (vo-all default), dezinger (0 default), dezinger-threshold (5000 integer), fbp-filter (parzen), bright-ratio (1), minus-log (True), flat-linear (False). These are separate from search geometry. Matching the reference command requires setting Dezinger to 5 explicitly; it is not silently enabled.

Do not mix angle-sweep flags into center sweeps or either sweep into full reconstruction. The current backend still emits lamino-search-width/step for fixed-angle jobs; these are redundant and do not mean that an angle search is executed.

find-center-start-row/end-row belong to built-in automatic center finding, not the displayed manual candidate slice. With a user-specified center and automatic-axis finding disabled they are unnecessary for this workflow. file-type standard and a flat-file-name equal to the input are optional explicit defaults. A different flat file requires an explicit flat-file-name.

In the installed reader, command-line center-search-step is multiplied by 2**binning when constructing the original-pixel center grid. Therefore binning=2 and center-search-step=4 produce a 16-original-pixel spacing. GUI values are in original pixels and are divided by this factor before calling TomocuPy. This rule must be verified separately for other installations.

For laminography, start-row/end-row describe the input projection-row read range; keep the full projection height. They must not be confused with lamino-start-row/end-row, which select reconstructed output planes. nsino is for trial planes only.
