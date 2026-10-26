# Segmentation timing and progress update — 1.0.1

Measured on the same NMC811_Rod data, GPU 1 (RTX 6000 Ada 48 GB), using the same checkpoint and unchanged segmentation/scoring operations. Twelve existing reconstructed slices were read; the original 2048-slice output was not overwritten.

Sample detector rows: 0, 256, 512, 768, 960, 1023, 1024, 1088, 1280, 1536, 1792, 2047.

Resident-model loading: 23.03 s, excluded from the per-image table. Twelve-image stage wall time: 32.52 s. Mean processing per image: 2.705 s.

| Operation | Mean per image (s) | Measured share |
|---|---:|---:|
| Generate label image and instance statistics | 0.8305 | 30.7% |
| SAM3 inference and mask transfer | 0.5521 | 20.4% |
| Scoring and mask NMS | 0.5101 | 18.9% |
| Compress and save instance masks | 0.3349 | 12.4% |
| Pack instance masks | 0.2793 | 10.3% |
| Read and normalize image | 0.1785 | 6.6% |
| Write label TIFF | 0.0120 | 0.4% |
| Flush instance CSV | 0.0016 | 0.1% |
| Progress and log output | 0.0014 | 0.1% |

The largest measured operation is CPU label-image construction, followed by SAM3 inference, scoring/NMS, and mask packing/compression. The label TIFF write itself is small. These are sample measurements; they do not retroactively measure every internal operation of the earlier 97-minute full segmentation. The remaining fraction includes timing/status/CSV bookkeeping.

New runs save `segmentation/timings.csv`, `segmentation/timing_summary.json`, and per-phase timing totals in `analysis_summary.json`. Existing masks, confidence scores, overlap handling, and center-search filtering are unchanged.

## Display changes

- Explicit stage 1/3 center search, stage 2/3 reconstruction, stage 3/3 segmentation (or two stages when Segment All is off).
- Segmentation states: **Reconstruction completed — segmenting all data**.
- Center-search rounds are shown only during center search; image counts are shown during reconstruction/segmentation.
- Progress percentages refer to the current round/stage. Entire-job completion is shown only after saving the final report.
- Total elapsed time uses HH:MM:SS for runs over an hour.
- Score plots automatically follow the current search interval(s), including local y scaling. Global view remains available.
- Global cumulative scores and Gaussian filtering still use all measured points; display zoom does not discard or recompute them.

## Validation

- 57 automated tests passed, including actual widget text/counters, segmentation-output equivalence, detailed timings, current-range/global-view switching and unchanged default planning.
- A real-data source GUI center search, five-image reconstruction/segmentation, new progress-event checks and Safe Stop passed.
- Saved-log display replay showed an x range of 1016.41–1032.34 px while retaining all 8189 global filtered curve points.
- Original input size and modification time were unchanged.

- Packaged EXE manual reconstruction and automatic five-image reconstruction/segmentation passed. The native worker emitted phase=completed with 5/5 images at the end.
