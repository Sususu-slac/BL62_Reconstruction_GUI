# UI update 1.2.5

Advanced Settings is now a compact right-aligned gear. The shared candidate tab is Parameter Optimization, and its action is Apply Selected Parameter.

TIFF enumeration, rewrite checksums and stable-write checks now run outside the GUI thread. A final asynchronous scan completes before the job reports Completed. Each scan merges the result list once and requests one preview. TIFF decoding and the existing 1–99 percentile display normalization run in the background. At most one preview per view is active, with only the latest pending selection retained. Resetting a view invalidates stale work. Console chunks are appended to the log together.

No reconstruction geometry or preprocessing parameters changed in this update.

Validation: 90 unit/GUI checks passed, including asynchronous final-scan delivery, UI responsiveness during a slow filesystem scan, batch candidates, stale-preview cancellation, browsing, both geometries and preprocessing.

A native Windows offscreen replay used 400 existing 2048×2048 float32 candidate TIFFs from the Bulk_Pristine comparison. Old per-file preview/list processing blocked the event loop for 130.315 seconds. The new single-batch UI update took 0.01834 seconds, with a maximum observed event-loop heartbeat gap of 34.04 ms during this replay. Preview loading completed. This measures UI updates, not reconstruction throughput or every possible source of a stall.

Raw evidence: test_outputs/responsive_ui_20261104/benchmark_before.json, benchmark_after.json, tests.log.
