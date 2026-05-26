Code assessing the connectivity from the anterior insulae to a range of frontal areas using rs-fMRI and movie-watching fMRI in individuals that self-report having had a traumatic brain injury and those that haven't

Author: Toby Engelking

# Cam-CAN Anterior Insula Connectivity Pipeline

This repository contains preprocessing and functional connectivity analysis pipelines used to extract seed-based and connectivity measures from Cam-CAN fMRI data, focusing on the anterior insulae across Movie and Rest conditions.

The pipeline supports multiple preprocessing variants, including:

- Movie task fMRI
- Resting-state fMRI
- Global Signal Regression (GSR)
- No Global Signal Regression
- Vectorised and iterative connectivity implementations

---

## Overview

The goal of this project is to characterise functional connectivity patterns of salience and control-related regions, with a focus on:

- Anterior insula (left and right seeds)
- Mid insula
- Ventrolateral prefrontal cortex (vlPFC)
- Orbitofrontal cortex (OFC)
- Anterior cingulate cortex (ACC)

Connectivity is computed using the Schaefer 200-parcel atlas (Yeo 7-network parcellation), with time series extracted from preprocessed Cam-CAN fMRI data.

---

## Data

This pipeline was developed for use with the **Cam-CAN dataset**.

Raw data is not included in this repository.

Access to Cam-CAN data must be obtained through a request to the Cam-CAN team available online.

---

## Pipeline Features

### Preprocessing
- NiftiLabelsMasker-based parcel extraction (nilearn)
- Motion regression
- Optional global signal regression (compCor-based)
- Temporal filtering (high-pass: 0.008 Hz)
- Z-scoring of time series

### Connectivity estimation
- ROI × ROI Pearson correlation
- Fisher Z-transformation
- Seed-based connectivity extraction (Anterior Insula L/R)
- Group-average connectivity matrices

---

## Folder Structure
scripts/
movie/
rest/

config/
paths.py
pipeline_settings.py

outputs/
(generated results; not tracked in git)

data/
(not included)

README.md
requirements.txt
.gitignore

## Configuration

All file paths and key parameters are centralised in the `config/` folder:

- `config/paths.py` → dataset locations and output directories
- `config/pipeline_settings.py` → preprocessing and connectivity parameters

Example:

```python
from config.paths import data_root, tbi_csv, get_output_dir
from config.pipeline_settings import TR, HIGH_PASS

## Requirements
numpy
pandas
nibabel
nilearn
scipy
matplotlib