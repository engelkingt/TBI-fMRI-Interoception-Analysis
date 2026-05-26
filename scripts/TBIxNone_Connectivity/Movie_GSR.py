# ============================================================
# AI connectivity extraction script — MOVIE task (Cam-CAN)
# Seed-based + full ROI connectivity pipeline
# GLOBAL SIGNAL REGRESSION INCLUDED (compSignal-based)
# ============================================================

import os
import time
import numpy as np
import pandas as pd
import nibabel as nib
import scipy.io as sio

from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker
from nilearn.signal import clean


# ============================================================
# CONFIG — ALL PATHS IN ONE PLACE (EDIT HERE ONLY)
# ============================================================
CONFIG = {
    "data_root": r"\\cbsu\data\Group\Camcan\cc700\mri\pipeline\release004\data_fMRI",
    "tbi_csv": r"U:\Work\Projects\Cam-CAN\Data Analysis\Data\rsFMRI\TBI.csv",
    "output_dir": r"U:\Work\Projects\Cam-CAN\Data Analysis\rsFMRI - Analyses\Attempt 8",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

output_csv_L = os.path.join(CONFIG["output_dir"], "AI_L_connectivity_movie_8.csv")
output_csv_R = os.path.join(CONFIG["output_dir"], "AI_R_connectivity_movie_8.csv")
output_matrix = os.path.join(CONFIG["output_dir"], "AI_connectivity_matrix_movie_8.npy")


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
# STEP 5 — Prepare output containers
# ------------------------------------------------------------
results_L = []
results_R = []
all_matrices = []


# ------------------------------------------------------------
# STEP 6 — Loop over participants (fMRI extraction + denoising)
# ------------------------------------------------------------
total_participants = len(tbi_info)
start_time = time.time()

for i, row in enumerate(tbi_info.itertuples(), start=1):
    participant_id = str(row.ParticipantID)
    tbi_status = row.TBI
    print(f"[{i}/{total_participants}] Processing {participant_id}...")

    movie_norm_folder = os.path.join(
        CONFIG["data_root"],
        "aamod_norm_write_dartel_00001",
        participant_id,
        "Movie"
    )

    movie_rp_folder = os.path.join(
        CONFIG["data_root"],
        "aamod_realignunwarp_00001",
        participant_id,
        "Movie"
    )

    comp_file = os.path.join(
        CONFIG["data_root"],
        "aamod_compSignal_00001",
        participant_id,
        "Movie",
        "compSignal.mat"
    )

    if not os.path.exists(movie_norm_folder) or not os.path.exists(movie_rp_folder) or not os.path.exists(comp_file):
        print(f"{participant_id}: Missing folder/file — skipping")
        continue

    nii_files = [f for f in os.listdir(movie_norm_folder) if f.lower().startswith("swauf") and f.endswith(".nii")]
    if not nii_files:
        print(f"{participant_id}: No functional file — skipping")
        continue

    func_file = os.path.join(movie_norm_folder, nii_files[0])

    try:
        func_img = nib.load(func_file)
        masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False)
        parcel_ts = masker.fit_transform(func_img)
    except Exception as e:
        print(f"{participant_id}: fMRI extraction error — skipping ({e})")
        continue


    # --------------------------------------------------------
    # Confound processing (motion + global signal)
    # --------------------------------------------------------
    try:
        rp_files = [f for f in os.listdir(movie_rp_folder) if f.lower().startswith("rp_fmr") and f.endswith(".txt")]
        if not rp_files:
            print(f"{participant_id}: Missing motion file — skipping")
            continue

        motion = np.loadtxt(os.path.join(movie_rp_folder, rp_files[0]))
        compTC = sio.loadmat(comp_file)["compTC"]
        global_signal = np.mean(compTC, axis=1, keepdims=True)

        confounds = np.hstack([motion, global_signal])

        if confounds.shape[0] != parcel_ts.shape[0]:
            print(f"{participant_id}: Timepoint mismatch — skipping")
            continue

    except Exception as e:
        print(f"{participant_id}: confound error — skipping ({e})")
        continue


    # --------------------------------------------------------
    # Time series cleaning (motion + GSR)
    # --------------------------------------------------------
    try:
        parcel_ts_clean = clean(
            parcel_ts,
            confounds=confounds,
            standardize="zscore_sample",
            t_r=2.0,
            high_pass=0.008,
            low_pass=None
        )
    except Exception as e:
        print(f"{participant_id}: cleaning error — skipping ({e})")
        continue

    roi_ts = parcel_ts_clean[:, [roi_indices[r] for r in roi_names]]

    if np.isnan(roi_ts).any():
        print(f"{participant_id}: NaNs detected — skipping")
        continue


    # --------------------------------------------------------
    # Connectivity estimation (ROI × ROI)
    # --------------------------------------------------------
    corr_matrix = np.corrcoef(roi_ts.T)
    z_matrix = np.arctanh(np.clip(corr_matrix, -0.999999, 0.999999))
    np.fill_diagonal(z_matrix, 1.0)

    all_matrices.append(z_matrix)

    seed_L_idx = roi_names.index("Anterior_Insula_L")
    seed_R_idx = roi_names.index("Anterior_Insula_R")

    results_L.append([participant_id, tbi_status] + z_matrix[seed_L_idx, :].tolist())
    results_R.append([participant_id, tbi_status] + z_matrix[seed_R_idx, :].tolist())

    elapsed = (time.time() - start_time) / 60
    print(f"{participant_id}: complete | elapsed ~{elapsed:.1f} min")


# ------------------------------------------------------------
# STEP 7 — Save outputs
# ------------------------------------------------------------
df_L = pd.DataFrame(results_L, columns=["ParticipantID", "TBI"] + roi_names)
df_R = pd.DataFrame(results_R, columns=["ParticipantID", "TBI"] + roi_names)

df_L.to_csv(output_csv_L, index=False)
df_R.to_csv(output_csv_R, index=False)

if all_matrices:
    group_matrix = np.mean(all_matrices, axis=0)
    np.save(output_matrix, {"matrix": group_matrix, "roi_order": roi_names})

print("===================================================")
print("Finished — Attempt 8 (GSR INCLUDED, FULL PARCEL SET)")
print("Outputs saved:")
print(output_csv_L)
print(output_csv_R)
print(output_matrix)
print("ROI order matches CSV column order EXACTLY")
print("===================================================")