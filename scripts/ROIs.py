import numpy as np
import nibabel as nib
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.image import new_img_like, resample_to_img
from nilearn import plotting

# -----------------------------
# 1) Load local AAL3 reference atlas
# -----------------------------
# Replace this with your local path via config/paths.py
aal3_file = "/path/to/AAL3/ROI_MNI_V7_1mm.nii"
ref_img = nib.load(aal3_file)

# -----------------------------
# 2) Fetch Schaefer 200 atlas (7 networks)
# -----------------------------
schaefer = fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=1)
schaefer_img = schaefer.maps
schaefer_labels = schaefer.labels

# -----------------------------
# 3) Resample Schaefer atlas to match AAL3 space
# -----------------------------
schaefer_resampled = resample_to_img(schaefer_img, ref_img, interpolation='nearest')
schaefer_data = schaefer_resampled.get_fdata()

# -----------------------------
# 4) Define all parcels to plot (updated)
# -----------------------------
parcels = {
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

# -----------------------------
# 5) Loop through parcels and plot each individually
# -----------------------------
for display_name, parcel_label in parcels.items():
    try:
        idx = schaefer_labels.index(parcel_label)
    except ValueError:
        print(f"Parcel {parcel_label} not found. Skipping.")
        continue

    # Create binary mask for the parcel
    mask_data = (schaefer_data == (idx + 1)).astype(np.int16)
    parcel_img = new_img_like(ref_img, mask_data)

    # Plot parcel on its own
    plotting.plot_stat_map(
        parcel_img,
        title=f"{display_name}: {parcel_label}",
        display_mode='ortho',
        colorbar=True,
        threshold=0.5
    )

# -----------------------------
# 6) Show all plots at the end
# -----------------------------
plotting.show()