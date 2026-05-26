# ============================================================
# Pipeline-wide settings (consistent across Movie/Rest scripts)
# ============================================================

TR = 2.0

# Filtering
HIGH_PASS = 0.008
LOW_PASS = None

# Connectivity settings
CORR_METHOD = "correlation"

# Atlas
ATLAS = "schaefer_2018"
N_ROIS = 200
YE0_NETWORKS = 7

# Denoising options
STANDARDIZE = "zscore_sample"

# Global signal regression toggle
USE_GSR = False  # flip True/False per script if needed