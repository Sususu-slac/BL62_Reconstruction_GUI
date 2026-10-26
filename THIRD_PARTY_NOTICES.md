BL62 Reconstruction uses Meta SAM3 for text-prompt particle segmentation.
SAM3 source and model materials are subject to the SAM License, reproduced in
licenses/SAM_LICENSE. Checkpoint files are external and are not redistributed.

The Windows bundle includes SAM3 inference source from
https://github.com/facebookresearch/sam3 at commit
4bf4a8aaff2ab34789731810eed45e6a61a73cfd, without modifications.

The rotation-center planner/scoring package is copied from the validated local
03_multithreshold_stability method. Its default cumulative global filtering,
confidence reward and full-resolution NMS are preserved.

Other runtime packages retain their respective licenses, which are included
by their distributions. TomocuPy and ASTRA run in the external reconstruction
environment; PyTorch and SAM3 run in the external SAM3 environment.
