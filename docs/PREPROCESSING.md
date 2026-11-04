# Preprocessing — 1.2.2

Manual reconstruction: use the right-aligned **Advanced Settings** gear in Reconstruction parameters.
Automatic analysis: open **Settings → Advanced Settings**. Both interfaces share the same saved values.

- Remove stripes: Vo-all (default), None, Fourier-wavelet, Titarenko.
- Dezinger size: Off by default; enter 5 to match the reference command. Threshold defaults to 5000.
- FBP filter: Parzen by default.
- Bright ratio: 1.0; exposure of flat frames divided by projection exposure.
- Minus-log: enabled.
- Flat-linear: disabled; interpolation requires at least three flat frames.

Settings are applied to manual trials, automatic center candidates and final reconstruction. ASTRA uses the same TomocuPy GPU correction functions before reconstruction. FBP filter selection applies to analytic methods; Cosine2 is available only through TomocuPy. Settings and effective preprocessing are recorded with each job.

Changes affect new jobs; reopen the latest desktop version to load the new controls.
