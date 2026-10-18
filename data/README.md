# Data Directory

This directory holds the raw and sampled dataset files from the **PrimeVul** security benchmark.

## Dataset Details
- **Source:** Hugging Face `colin/PrimeVul`
- **Total Samples:** ~20,000 code snippets (C/C++ & Python)
- **Label Column:** `target` (`0` = Safe, `1` = Vulnerable)
- **Code Column:** `func`
- **Class Imbalance:** ~97% safe vs ~3% vulnerable

## Acquiring the Data
Run the included download script to fetch and sample the dataset:
```bash
python data/download.py
```
This generates `data/primevul_sample.csv` (excluded from git due to file size).
