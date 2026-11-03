# Fixed-angle laminography analysis

On Intelligent Data Analysis (Test), **Laminography** is unchecked by default. Leave it unchecked for tomography. Check it to enter the tilt angle in degrees; cancelling the dialog clears the checkbox. The entered angle is applied to candidate reconstructions, rotation-center search and final reconstruction, and is recorded in the job configuration.

Center-search candidates use LineRec. The final reconstruction uses laminography FourierRec or LineRec; other tomography-only choices fall back to laminography LineRec. The angle is fixed; automatic angle estimation is not implemented.

Input HDF5 files can contain `exchange/theta` in either acquisition geometry, so the presence of theta alone does not select tomography. Select the known acquisition geometry explicitly.

Validation covers GUI controls, job configuration, existing backend command coverage and Windows bundle startup. It does not establish reconstruction quality on every laminography dataset.
