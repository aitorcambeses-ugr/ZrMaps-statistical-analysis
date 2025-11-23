# EXECUTION PROTOCOL & OUTPUT INTERPRETATION GUIDE
### ZrMaps-statistical-analysis

This document explains how to execute all processing scripts included in the repository and how to interpret all generated output files. It serves as the official workflow reference for running micro-XRF, EPMA/SEM–EDS-derived chemical map analysis for zircon detection.

---

# 0. Overview of the Workflow

The complete workflow processes raw JPG chemical maps and produces:

## 0.1 Input data
You must provide **10 JPG element maps** per sample:
- Si, Ti, Al, Fe, Mg, Ca, Na, K, P, Zr

Stored inside:
`SAMPLE/`

## 0.2 Main processing steps
1. JPG → TSV conversion  
2. FRAC chemical ratios + KMeans (15-phase classification)  
3. Statistical Zr anomaly tests  
4. DBSCAN + HDBSCAN clustering  

## 0.3 Output folders & interpretation

### A) JPG → TSV conversion
Folder: `SAMPLE_mapas_tsv/`
- One TSV file per element
- Raw intensity pixel matrix

### B) FRAC phase classification
Folder: `SAMPLE_FRAC_clusters/`
- FRAC maps
- KMeans phase classification  
- Legend with color-coded phases

### C) Statistical Zr anomaly tests
Folders:
- `SAMPLE_Zr_IQR_anomaly/`
- `SAMPLE_Zr_MAD_anomaly/`
- `SAMPLE_Zr_Zscore_anomaly/`
- `SAMPLE_Zr_GMM_anomaly/`
- `SAMPLE_Zr_Adaptive_anomaly/`
- `SAMPLE_Zr_LocalMAD_anomaly/`

Each folder contains:
- Linear anomaly map  
- Log10(Zr) anomaly map  
- histogram.png  
- *_matrix.tsv (0–4 zircon scale)  
- *_stats.txt  

Zircon classification scale:
- 0 = no zircon  
- 1 = dubious  
- 2 = possible zircon  
- 3 = highly probable zircon  
- 4 = zircon  

Report:
`SAMPLE_Zr_alltests_report.pdf`

### D) DBSCAN / HDBSCAN clustering
Folders:
- `SAMPLE_Zr_DBSCAN/`
- `SAMPLE_Zr_HDBSCAN/`

Contains:
- Cluster maps  
- TSV cluster matrices  
- Optional PDF summary  

---

# 1. macOS (Terminal)

## 1.1 Initial Setup

```bash
mkdir -p ~/Desktop/micro_XRF
```

Copy scripts:
- images_to_tsv_auto_delete.py  
- frac_classification_auto.py  
- zircon_zr_alltests_FULL.py  
- zircon_zr_DBSCAN_HDBSCAN.py  

Install dependencies:

```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install numpy matplotlib scikit-learn hdbscan pillow openpyxl
```

If HDBSCAN fails:

```bash
brew install cmake gcc llvm
HDBSCAN_NO_AVX=1 python3 -m pip install hdbscan
```

## 1.2 Full Processing (per sample)

```bash
SAMPLE="TOB_7_100m_10TP"
mkdir -p ~/Desktop/micro_XRF/"$SAMPLE"
cd ~/Desktop/micro_XRF
python3 images_to_tsv_auto_delete.py "$SAMPLE"
python3 frac_classification_auto.py "$SAMPLE"
python3 zircon_zr_alltests_FULL.py "$SAMPLE"
python3 zircon_zr_DBSCAN_HDBSCAN.py "$SAMPLE"
ls "$SAMPLE"
```

## 1.3 Quick Re-run

```bash
SAMPLE="TOB_7_100m_10TP"
cd ~/Desktop/micro_XRF
rm -rf "$SAMPLE/${SAMPLE}_FRAC_clusters"
python3 frac_classification_auto.py "$SAMPLE"
rm -rf "$SAMPLE"/*Zr*_anomaly
rm -f "$SAMPLE/${SAMPLE}_Zr_alltests_report.pdf"
python3 zircon_zr_alltests_FULL.py "$SAMPLE"
rm -rf "$SAMPLE/${SAMPLE}_Zr_DBSCAN"
rm -rf "$SAMPLE/${SAMPLE}_Zr_HDBSCAN"
python3 zircon_zr_DBSCAN_HDBSCAN.py "$SAMPLE"
```

---

# 2. Windows (CMD)

## 2.1 Requirements
Install Python 3 (with Add to PATH)

```cmd
pip install --upgrade pip setuptools wheel
pip install numpy matplotlib scikit-learn hdbscan pillow openpyxl
```

## 2.2 Full Processing

```cmd
mkdir "%USERPROFILE%\Desktop\micro_XRF"
cd "%USERPROFILE%\Desktop\micro_XRF"
set SAMPLE=test
python images_to_tsv_auto_delete.py "%SAMPLE%"
python frac_classification_auto.py "%SAMPLE%"
python zircon_zr_alltests_FULL.py "%SAMPLE%"
python zircon_zr_DBSCAN_HDBSCAN.py "%SAMPLE%"
dir "%SAMPLE%"
```

## 2.3 Quick Re-run

```cmd
set SAMPLE=test
cd "%USERPROFILE%\Desktop\micro_XRF"
rmdir /S /Q "%SAMPLE%\%SAMPLE%_FRAC_clusters"
python frac_classification_auto.py "%SAMPLE%"
rmdir /S /Q "%SAMPLE%\%SAMPLE%_Zr_*_anomaly"
del "%SAMPLE%\%SAMPLE%_Zr_alltests_report.pdf"
python zircon_zr_alltests_FULL.py "%SAMPLE%"
rmdir /S /Q "%SAMPLE%\%SAMPLE%_Zr_DBSCAN"
rmdir /S /Q "%SAMPLE%\%SAMPLE%_Zr_HDBSCAN"
python zircon_zr_DBSCAN_HDBSCAN.py "%SAMPLE%"
```

---

# 3. Windows (PowerShell)

## 3.1 Full Processing

```powershell
$env:SAMPLE="test"
mkdir "$env:USERPROFILE\Desktop\micro_XRF\$env:SAMPLE"
cd "$env:USERPROFILE\Desktop\micro_XRF"
python images_to_tsv_auto_delete.py $env:SAMPLE
python frac_classification_auto.py $env:SAMPLE
python zircon_zr_alltests_FULL.py $env:SAMPLE
python zircon_zr_DBSCAN_HDBSCAN.py $env:SAMPLE
ls "$env:SAMPLE"
```

## 3.2 Quick Re-run

```powershell
$env:SAMPLE="test"
cd "$env:USERPROFILE\Desktop\micro_XRF"
Remove-Item "$env:SAMPLE\${env:SAMPLE}_FRAC_clusters" -Recurse -Force
python frac_classification_auto.py $env:SAMPLE
Remove-Item "$env:SAMPLE\${env:SAMPLE}_Zr_*_anomaly" -Recurse -Force
Remove-Item "$env:SAMPLE\${env:SAMPLE}_Zr_alltests_report.pdf" -Force
python zircon_zr_alltests_FULL.py $env:SAMPLE
Remove-Item "$env:SAMPLE\${env:SAMPLE}_Zr_DBSCAN" -Recurse -Force
Remove-Item "$env:SAMPLE\${env:SAMPLE}_Zr_HDBSCAN" -Recurse -Force
python zircon_zr_DBSCAN_HDBSCAN.py $env:SAMPLE
```

---

# END OF DOCUMENT
