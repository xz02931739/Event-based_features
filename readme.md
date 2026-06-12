# Event-based_features — SpO2 Feature Extraction and Event-Based Metrics

This repository extracts clinically relevant features from overnight SpO2 signals, including time-domain, frequency-domain, non-linear, and event-based features (overall and stratified by sleep stage). It consumes packed .npy files per subject and outputs CSVs ready for downstream analysis or modeling.

## Repository layout

```
get_features_time.py                    # Time-domain features (incl. sleep-stage summaries)
get_features_fre.py                     # Frequency-domain features (PSD, spectral entropy, moments)
get_features_nonlinear.py               # Nonlinear features (ApEn, SampleEn, LZC)
get_features_event.py                   # Event-based features and ODI-like rate
get_features_event_based_stages.py      # Event-based features split by sleep stages
original_frame.csv                      # Subject list with labels (nsrrid, level)
saved/                                  # Generated feature CSVs are written here
small_shhs1_spo2_pack/all/              # Example SpO2 packs (shhs1-<id>.npy)
utils/                                  # Calculation helpers (time, freq, nonlinear, events)
```

## Data assumptions

- Each subject is represented by one .npy file at `small_shhs1_spo2_pack/all/shhs1-<nsrrid>.npy` with a Python dict containing:
	- `data`: 1D SpO2 array (integers/percent) sampled at `fs` Hz
	- `label`: 1D array of sleep stage codes per 30s epoch (0=wake, 1/2=light, 3=deep, 5=REM)
	- `fs`: Sampling frequency (Hz)
- `original_frame.csv` includes at least two columns:
	- `nsrrid`: subject identifier matching the file naming convention above
	- `level`: target label for classification (3- or 4-class setups; not used by feature calc, but preserved in outputs)

## Installation

Use Python 3.8+ on Windows. Create a virtual environment and install dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you prefer a manual install, the core packages are: numpy, pandas, scipy, tqdm, antropy, lempel-ziv-complexity, scikit-learn, matplotlib.

## Usage

All scripts expect to be run from the repository root. They read `original_frame.csv` and the `small_shhs1_spo2_pack/all/` folder, then write feature CSVs into `saved/`.

Run each feature extractor independently:

```powershell
# Time-domain features (incl. CT90/88/86 and stage-wise mean/min/max)
python .\get_features_time.py

# Frequency-domain features (band power 0.01–0.03 Hz, spectral entropy, PSD moments)
python .\get_features_fre.py

# Nonlinear features (ApEn, SampleEn, Lempel–Ziv complexity)
python .\get_features_nonlinear.py

# Event-based features (area, duration, EFR/ERR/EBS, plus calculated ODI)
python .\get_features_event.py

# Event-based features split by sleep stage (wake/light/deep/REM)
python .\get_features_event_based_stages.py
```

Outputs will appear as CSVs:

- `saved/features_time.csv`
- `saved/features_frequency.csv`
- `saved/features_nonlinear.csv`
- `saved/features_event.csv`
- `saved/features_event_splited.csv`

> Tip: You can run only the scripts you need. Each script includes a `__main__` that reads from `original_frame.csv` and writes its own output.

## What each script computes

### Time-domain (`get_features_time.py`)
- Preprocessing: values <= 25 are treated as missing and linearly interpolated
- Summary statistics: `Variance_SpO2`, `Skewness_SpO2`, `Kurtosis_SpO2`, overall `Mean/Minimum/Maximum`
- Hypoxic burden: cumulative time below thresholds (seconds) `CT90`, `CT88`, `CT86`
- Stage-wise summaries (mean/min/max) for wake, light (N1+N2), deep (N3), REM

Columns include: `CT90, CT88, CT86, Variance_SpO2, Skewness_SpO2, Kurtosis_SpO2, Wake_mean/min/max, Light_mean/min/max, Deep_mean/min/max, REM_mean/min/max, Mean, Minimum, Maximum, level`.

### Frequency-domain (`get_features_fre.py`)
- Welch PSD (300 s window, 75% overlap)
- `PSD`: band power in 0.01–0.03 Hz
- `Peak_fs`: PSD peak within that band
- `Spectralentropy_PSD`: spectral entropy of PSD
- PSD statistical moments: mean, variance, skewness, kurtosis

Columns include: `PSD, Peak_fs, Spectralentropy_PSD, PSD_mean, PSD_Variance, PSD_Skewness, PSD_Kurtosis, level`.

### Nonlinear (`get_features_nonlinear.py`)
- Approximate entropy (`apen`)
- Sample entropy (`sample_en`)
- Lempel–Ziv complexity (`lz_complexity`)

Columns include: `apen, sample_en, lz_complexity, level`.

### Event-based overall (`get_features_event.py`)
- Event detection: find desaturation blocks around local minima using slope-based arms (±0.5 min), deduplicate overlaps, revise nadir
- Per-event metrics: area, duration, fall height, EFR (fall rate), recover height, ERR (rise rate), EBS (overall rate)
- Aggregation: mean of each metric weighted by event frequency per minute of sleep; derived ODI-like rate (events/hour)

Columns include: `Area, Duration, Fall_height, EFR, Rise_height, ERR, EBS, Calculated_ODI, Total_number_events, level`.

### Event-based by sleep stage (`get_features_event_based_stages.py`)
- Assign each detected event to the most frequent sleep stage overlapping its duration
- Compute the same event metrics as above, separately for Wake, Light, Deep, REM, weighted by time spent in that stage

Columns include for each stage prefix (`Wake_`, `Light_`, `Deep_`, `REM_`): `area, duration, fall_height, EFR, recover_height, ERR, EBS`, plus `level`.

## Customization

- Data locations: change `pack_dir` or `original_frame.csv` path in each script’s `__main__` if your files live elsewhere
- Thresholding: the interpolation threshold (default 25) is set in `utils/balanced_clinical_tools.py::fill_nan_with_threshold`
- Frequency band: adjust `FrequencyDomain_tool(...).band_power([0.01, 0.03])`
- Sleep stage mapping: stage codes are handled in `utils/cal_timedomain.py` and the stage assignment logic in `get_features_event_based_stages.py`

## Example: minimal end‑to‑end run

1) Ensure you have:
	 - `original_frame.csv` with columns `nsrrid, level`
	 - Matching `.npy` packs under `small_shhs1_spo2_pack/all/` containing `data`, `label`, `fs`
2) Activate your environment and run a script, e.g. time features:


## Notes

- Sleep epoch is 30s; several computations convert counts to minutes/hours accordingly
- All scripts preserve the input `level` column to simplify downstream classification experiments
- Utility functions for dataset splitting/scaling are available in `utils/split_5fd.py` and `utils/cm_self.py`

