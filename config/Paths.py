# ============================================================
# Centralised file paths for Cam-CAN AI connectivity project
# ============================================================

import os

# -----------------------------
# DATA (Cam-CAN shared drive)
# -----------------------------
data_root = r"\\cbsu\data\Group\Camcan\cc700\mri\pipeline\release004\data_fMRI"

# -----------------------------
# PARTICIPANT METADATA
# -----------------------------
tbi_csv = r"U:\Work\Projects\Cam-CAN\Data Analysis\Data\rsFMRI\TBI.csv"

# -----------------------------
# OUTPUT ROOT DIRECTORY
# -----------------------------
output_root = r"U:\Work\Projects\Cam-CAN\Data Analysis\rsFMRI - Analyses"

# -----------------------------
# ATLAS / ROI FILES
# -----------------------------
aal3_file = r"U:\Work\Projects\Cam-CAN\Data Analysis\rsFMRI - Analyses\Attempt 8\ROIs\AAL3\ROI_MNI_V7_1mm.nii"

# -----------------------------
# OPTIONAL: derived paths helper
# -----------------------------
def get_output_dir(attempt_name: str):
    return os.path.join(output_root, attempt_name)