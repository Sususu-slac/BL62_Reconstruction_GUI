# BL62 Reconstruction 1.2.7

Manual laminography full reconstruction now follows the output height calculated by TomoCuPy for the selected tilt and binning. A 2048-row detector at 30 degrees and original resolution produces all 2366 output slices instead of stopping at 2048. Detector rows used for center trials remain in detector coordinates; explicit cropped output ranges remain available.

Manual center trials include aligned endpoints: center ±1 pixel at a 0.25-pixel step generates nine candidates. A process-local adapter changes only the manual center grid; installed TomoCuPy files and the automatic search pipeline are unchanged.

The background result scanner waits for complete TIFF headers and pixel blocks before offering a preview.

Validation: independent manual trial and full reconstruction workflows were run through the reconstruction GUI and MomoBot at center 1070.5, tilt +30 degrees, LineRec, Vo-all and Parzen. Each produced 2366 float32 2048×2048 TIFFs and nine trial images, all bitwise identical to the preserved reference. 118 focused/existing manual and preview regression checks passed. This validates the stated manual workflow; it is not a new automatic-search or reconstruction-quality benchmark. Source data, private chats and workstation logs are not included.

The Windows installer and portable application use the existing scientific Python environments; those environments and model weights remain external. Reopen an existing application after updating.
