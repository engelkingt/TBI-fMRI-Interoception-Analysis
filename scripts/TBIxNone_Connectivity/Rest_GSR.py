# ============================================================
# AI connectivity extraction script — REST task (Cam-CAN)
# Seed-based + full ROI connectivity pipeline
# Resting-state (WITH GLOBAL SIGNAL REGRESSION)
# FINAL PARCEL SET (Corrected Schaefer 200 mapping)
# ============================================================

import os
import time
import numpy as np
import pandas as pd
import nibabel as nib
import scipy.io as sio
import matplotlib.pyplot as plt

from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker
from nilearn.signal import clean


# ============================================================
# CONFIG — CENTRAL PATH MANAGEMENT
# ============================================================
CONFIG = {
    "data_root": r"\\cbsu\data\Group\Camcan\cc700\mri\pipeline\release004\data_fMRI",
    "tbi_csv": r"U:\Work\Projects\Cam-CAN\Data Analysis\Data\rsFMRI\TBI.csv",
    "output_dir": r"U:\Work\Projects\Cam-CAN\Data Analysis\rsFMRI - Analyses\Attempt 8",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

output_csv_L = os.path.join(CONFIG["output_dir"], "AI_L_connectivity_rest_GSR.csv")
output_csv_R = os.path.join(CONFIG["output_dir"], "AI_R_connectivity_rest_GSR.csv")
output_matrix = os.path.join(CONFIG["output_dir"], "AI_connectivity_matrix_rest_GSR.npy")


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
# STEP 4 — FINAL ROI DEFINITIONS (Corrected Schaefer parcels)
# ------------------------------------------------------------
roi_labels = {
    "Anterior_Insula_L": "7Networks_LH_SalVentAttn_FrOperIns_1",
    "Anterior_Insula_R": "7Networks_RH_SalVentAttn_FrOperIns_2",
    "Mid_Insula_L": "7Networks_LH_SalVentAttn_ParOper_3",
    "Mid_Insula_R": "7Networks_RH_SalVentAttn_FrOperIns_3",

    "vlPFC_L": "7Networks_LH_Cont_PFCl_3",
    "vlPFC_R": "7Networks_RH_Cont_PFCl_3",

    "OFC_L": "7Networks_LH_Limbic_OFC_1",
    "OFC_R": "7Networks_RH_Limbic_OFC_2",

    "ACC_L": "7Networks_LH_Default_PFC_5",
    "ACC_R": "7Networks_RH_Default_PFCdPFCm_2",
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
results_L = []
results_R = []
all_matrices = []

start_time = time.time()
total_participants = len(tbi_info)


# ------------------------------------------------------------
# STEP 6 — Loop participants (REST extraction + GSR)
# ------------------------------------------------------------
for i, row in enumerate(tbi_info.itertuples(), start=1):
    participant_id = str(row.ParticipantID)
    tbi_status = row.TBI
    print(f"[{i}/{total_participants}] {participant_id}")

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

    comp_file = os.path.join(
        CONFIG["data_root"],
        "aamod_compSignal_00001",
        participant_id,
        "Rest",
        "compSignal.mat"
    )

    # --------------------------------------------------------
    # Data availability check
    # --------------------------------------------------------
    missing = False
    for folder, name in [(rest_norm_folder, "norm"), (rest_rp_folder, "rp"), (comp_file, "compSignal")]:
        if not os.path.exists(folder):
            print(f"  Missing {name} data → skipping participant")
            missing = True
    if missing:
        continue

    # --------------------------------------------------------
    # Load functional data
    # --------------------------------------------------------
    nii_files = [f for f in os.listdir(rest_norm_folder) if f.startswith("swauf")]
    if not nii_files:
        print("  No functional NIfTI files found → skipping participant")
        continue

    func_img = nib.load(os.path.join(rest_norm_folder, nii_files[0]))

    masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False)
    parcel_ts = masker.fit_transform(func_img)

    # --------------------------------------------------------
    # Confounds (motion + global signal regression)
    # --------------------------------------------------------
    rp_files = [f for f in os.listdir(rest_rp_folder) if f.startswith("rp_")]
    if not rp_files:
        print("  No motion files found → skipping participant")
        continue

    motion = np.loadtxt(os.path.join(rest_rp_folder, rp_files[0]))
    compTC = sio.loadmat(comp_file)["compTC"]
    global_signal = np.mean(compTC, axis=1, keepdims=True)

    confounds = np.hstack([motion, global_signal])

    # --------------------------------------------------------
    # Cleaning (WITH GSR)
    # --------------------------------------------------------
    parcel_ts_clean = clean(
        parcel_ts,
        confounds=confounds,
        standardize="zscore_sample",
        t_r=2.0,
        high_pass=0.008
    )

    roi_ts = parcel_ts_clean[:, [roi_indices[r] for r in roi_names]]

    # --------------------------------------------------------
    # Connectivity estimation
    # --------------------------------------------------------
    corr = np.corrcoef(roi_ts.T)
    zmat = np.arctanh(np.clip(corr, -0.999999, 0.999999))
    np.fill_diagonal(zmat, 1.0)

    all_matrices.append(zmat)

    seed_L = roi_names.index("Anterior_Insula_L")
    seed_R = roi_names.index("Anterior_Insula_R")

    results_L.append([participant_id, tbi_status] + zmat[seed_L, :].tolist())
    results_R.append([participant_id, tbi_status] + zmat[seed_R, :].tolist())


# ------------------------------------------------------------
# STEP 7 — Save outputs
# ------------------------------------------------------------
if all_matrices:
    df_L = pd.DataFrame(results_L, columns=["ParticipantID", "TBI"] + roi_names)
    df_R = pd.DataFrame(results_R, columns=["ParticipantID", "TBI"] + roi_names)

    df_L.to_csv(output_csv_L, index=False)
    df_R.to_csv(output_csv_R, index=False)

    group_matrix = np.mean(all_matrices, axis=0)
    np.save(output_matrix, {"matrix": group_matrix, "roi_order": roi_names})

    print("DONE — Attempt 11 (REST, WITH GSR) outputs saved")

    # --------------------------------------------------------
    # STEP 8 — Group-level heatmap
    # --------------------------------------------------------
    plt.figure(figsize=(12, 10))
    im = plt.imshow(group_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Fisher Z")

    plt.xticks(range(len(roi_names)), roi_names, rotation=90, fontsize=12)
    plt.yticks(range(len(roi_names)), roi_names, fontsize=12)

    plt.title(
        "Anterior insula connectivity matrix (REST, with global signal regression)",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()

else:
    print("No participant data processed — cannot create connectivity matrix")