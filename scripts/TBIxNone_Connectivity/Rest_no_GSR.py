# ============================================================
# AI connectivity extraction script — REST task (Cam-CAN)
# Seed-based + full ROI connectivity pipeline
# Resting-state (NO GLOBAL SIGNAL REGRESSION)
# FINAL PARCEL SET (Anterior + Mid Insula separated)
# ============================================================

import os
import time
import numpy as np
import pandas as pd
import nibabel as nib

from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker
from nilearn.signal import clean
from nilearn.connectome import ConnectivityMeasure


# ============================================================
# CONFIG — CENTRAL PATH MANAGEMENT
# ============================================================
CONFIG = {
    "data_root": r"\\cbsu\data\Group\Camcan\cc700\mri\pipeline\release004\data_fMRI",
    "tbi_csv": r"U:\Work\Projects\Cam-CAN\Data Analysis\Data\rsFMRI\TBI.csv",
    "output_dir": r"U:\Work\Projects\Cam-CAN\Data Analysis\rsFMRI - Analyses\Attempt 8",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

output_csv_L = os.path.join(CONFIG["output_dir"], "AnteriorAI_L_connectivity_rest_NoGSR.csv")
output_csv_R = os.path.join(CONFIG["output_dir"], "AnteriorAI_R_connectivity_rest_NoGSR.csv")
output_matrix = os.path.join(CONFIG["output_dir"], "AI_connectivity_matrix_rest_NoGSR.npy")


# ------------------------------------------------------------
# STEP 2 — Load participant information
# ------------------------------------------------------------
tbi_info = pd.read_csv(CONFIG["tbi_csv"])
tbi_info.columns = ["ParticipantID", "TBI"]

# ------------------------------------------------------------
# STEP 3 — Load Schaefer 2018 atlas (200 parcels)
# ------------------------------------------------------------
schaefer = fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=2)
atlas_img = schaefer["maps"]
labels = schaefer["labels"][1:]  # drop background label

# ------------------------------------------------------------
# STEP 4 — FINAL ROI DEFINITIONS (Schaefer 200 mapping)
# ------------------------------------------------------------
roi_labels = {
    "Anterior_Insula_L": "7Networks_LH_SalVentAttn_FrOperIns_1",
    "Anterior_Insula_R": "7Networks_RH_SalVentAttn_FrOperIns_2",

    "Mid_Insula_L": "7Networks_LH_SalVentAttn_ParOper_3",
    "Mid_Insula_R": "7Networks_RH_SalVentAttn_FrOperIns_3",

    "OFC_L": "7Networks_LH_Limbic_OFC_1",
    "OFC_R": "7Networks_RH_Limbic_OFC_2",

    "vlPFC_L": "7Networks_LH_Cont_PFCl_3",
    "vlPFC_R": "7Networks_RH_Cont_PFCl_3",

    "ACC_L": "7Networks_LH_Default_PFC_5",
    "ACC_R": "7Networks_RH_Default_PFCdPFCm_2"
}


def get_label_index(target_label):
    matches = [i for i, lab in enumerate(labels) if target_label in lab]
    if len(matches) != 1:
        raise ValueError(f"Expected 1 match for '{target_label}', found {len(matches)}")
    return matches[0]


roi_indices = {roi: get_label_index(lbl) for roi, lbl in roi_labels.items()}
roi_names = list(roi_indices.keys())


# ------------------------------------------------------------
# STEP 5 — Prepare containers
# ------------------------------------------------------------
all_subject_ts = []
participant_ids = []
tbi_statuses = []

start_time = time.time()


# ------------------------------------------------------------
# STEP 6 — Extract ROI time series (REST condition)
# ------------------------------------------------------------
for i, row in enumerate(tbi_info.itertuples(), start=1):
    participant_id = str(row.ParticipantID)
    tbi_status = row.TBI
    print(f"[{i}/{len(tbi_info)}] Processing {participant_id}...")

    rest_norm_folder = os.path.join(
        CONFIG["data_root"],
        "aamod_norm_write_dartel_00001",
        participant_id,
        "Rest"
    )

    rest_rp_folder = os.path.join(
        CONFIG["data_root"],
        "aamod_realignunwarp_00001",
        participant_id,
        "Rest"
    )

    if not os.path.exists(rest_norm_folder) or not os.path.exists(rest_rp_folder):
        print(f"{participant_id}: Missing REST folders — skipping")
        continue

    nii_files = [f for f in os.listdir(rest_norm_folder)
                 if f.lower().startswith("swauf") and f.endswith(".nii")]

    if not nii_files:
        print(f"{participant_id}: No functional file — skipping")
        continue

    func_img = nib.load(os.path.join(rest_norm_folder, nii_files[0]))

    rp_files = [f for f in os.listdir(rest_rp_folder)
                if f.lower().startswith("rp_") and f.endswith(".txt")]

    if not rp_files:
        print(f"{participant_id}: No motion file — skipping")
        continue

    motion = np.loadtxt(os.path.join(rest_rp_folder, rp_files[0]))

    # --------------------------------------------------------
    # ROI time series extraction
    # --------------------------------------------------------
    masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False)
    parcel_ts = masker.fit_transform(func_img)

    # Motion-only denoising (NO GLOBAL SIGNAL REGRESSION)
    parcel_ts_clean = clean(
        parcel_ts,
        confounds=motion,
        standardize="zscore_sample",
        t_r=2.0,
        high_pass=0.008,
        low_pass=None
    )

    roi_ts = parcel_ts_clean[:, [roi_indices[r] for r in roi_names]]

    if np.isnan(roi_ts).any():
        print(f"{participant_id}: NaNs detected — skipping")
        continue

    all_subject_ts.append(roi_ts)
    participant_ids.append(participant_id)
    tbi_statuses.append(tbi_status)

elapsed = (time.time() - start_time) / 60
print(f"ROI extraction complete (~{elapsed:.1f} min)")


# ------------------------------------------------------------
# STEP 7 — Connectivity estimation (vectorised)
# ------------------------------------------------------------
connectivity_measure = ConnectivityMeasure(kind="correlation")
corr_matrices = connectivity_measure.fit_transform(all_subject_ts)

z_matrices = np.arctanh(np.clip(corr_matrices, -0.999999, 0.999999))
for z in z_matrices:
    np.fill_diagonal(z, 1.0)


# ------------------------------------------------------------
# STEP 8 — Seed-based outputs (Anterior Insula only)
# ------------------------------------------------------------
seed_L = roi_names.index("Anterior_Insula_L")
seed_R = roi_names.index("Anterior_Insula_R")

results_L = [[pid, tbi] + z[seed_L, :].tolist()
             for pid, tbi, z in zip(participant_ids, tbi_statuses, z_matrices)]

results_R = [[pid, tbi] + z[seed_R, :].tolist()
             for pid, tbi, z in zip(participant_ids, tbi_statuses, z_matrices)]

df_L = pd.DataFrame(results_L, columns=["ParticipantID", "TBI"] + roi_names)
df_R = pd.DataFrame(results_R, columns=["ParticipantID", "TBI"] + roi_names)

df_L.to_csv(output_csv_L, index=False)
df_R.to_csv(output_csv_R, index=False)


# ------------------------------------------------------------
# STEP 9 — Group-average connectivity matrix
# ------------------------------------------------------------
group_matrix = np.mean(z_matrices, axis=0)
np.save(output_matrix, {"matrix": group_matrix, "roi_order": roi_names})

print("===================================================")
print("Finished — REST pipeline (Attempt 10, NO GSR)")
print("Outputs saved:")
print(output_csv_L)
print(output_csv_R)
print(output_matrix)
print("ROI order matches CSV column order EXACTLY")
print("===================================================")