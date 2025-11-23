# ZrMaps-statistical-analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17690656.svg)](https://doi.org/10.5281/zenodo.17690656)

## Overview

**ZrMaps-statistical-analysis** is an open-source toolkit designed for the **statistical analysis, classification, and anomaly detection of chemical element maps** acquired using X-ray scanning techniques (e.g., micro-XRF, EPMA/WDS, SEM–EDS mapping, laboratory μXRF scanners, synchrotron XRF, etc.).

The project provides a set of **robust, reproducible, and scalable tools** to identify mineral phases—**especially zircon (ZrSiO₄)**—in large-area elemental maps (e.g., 10×10 cm thin sections scanned at 30–50 µm pixel size).  
The framework integrates both **classical univariate anomaly detection** (IQR, MAD, Z-score, adaptive percentile thresholds, local MAD) and **density-based clustering algorithms** (DBSCAN and HDBSCAN), offering complementary perspectives on Zr enrichment patterns.

Together, these methods deliver **fast and reliable identification of zircon grains** in complex geological matrices where traditional petrography is insufficient or too time-consuming.

---

## Scientific Motivation

Zircon is a cornerstone mineral in modern **geochronology**, **petrogenesis**, and **crustal evolution studies**.  
However, locating zircon grains in **large thin sections** or **drill-core slabs** mapped by micro-XRF or EPMA can be challenging due to:

- very low Zr concentrations in most rock-forming minerals  
- strong signal variability related to acquisition parameters  
- sub-pixel mixing effects at 30–50 μm grids  
- overlapping fluorescence lines (Zr–Kα, P–Kα, Si–Kα)

This toolkit provides **automated, statistically robust detection workflows** that minimise these limitations and allow high-confidence pre-screening before targeted geochemical analyses (EPMA, LA-ICP-MS, SHRIMP, TEM, atom-probe).

---

## Implemented Methods

### **1. Classical statistical anomaly detection**
Included in `zircon_zr_alltests_FULL.py`:

- **IQR thresholds**  
- **MAD (Median Absolute Deviation)**  
- **Z-score**  
- **Adaptive percentiles** (P85–P99.5)  
- **Local MAD**  
- **Gaussian Mixture Models (GMM)**  

Outputs include:

- anomaly maps (linear/log)  
- histograms with thresholds  
- combined multi-method panels  
- TSV/CSV outputs  
- multi-page PDF report  

---

### **2. Density-based clustering**
Included in `zircon_zr_DBSCAN_HDBSCAN.py`:

- DBSCAN + HDBSCAN  
- High-density anomaly core extraction  
- Zircon-likeness classification (0–4)  
- TSV cluster matrices  
- Optional PDF maps  

Especially effective for:

- isolated zircon grains  
- small zircon clusters in biotite/amphibole  
- zoning patterns  

---

### **3. Multi-element phase classification (FRAC)**

`frac_classification_auto.py` implements FRAC-based clustering using KMeans (k=15), enabling:

- contextualisation of zircon within host phases  
- approximate chemical partitioning  
- guidance for microanalysis  

---

## Repository Structure


ZrMaps-statistical-analysis/
│
├── src/
│   ├── images_to_tsv_auto_delete.py
│   ├── frac_classification_auto.py
│   ├── zircon_zr_alltests_FULL.py
│   └── zircon_zr_DBSCAN_HDBSCAN.py
│
├── data/      # optional example datasets
├── docs/      # manuals, reports
├── tests/     # future automated tests
│
├── README.md
└── LICENSE    # MIT License


---

## Funding Acknowledgement

This project has been developed within the framework of the research grant:

**PID2023.149105NA.I00**  
Funded by the **Spanish Ministry of Science, Innovation and Universities** (2023 call)  
Principal Investigator (PI): **Aitor Cambeses Torres**

---

## Citation (Text)

Cambeses, A. (2025). *ZrMaps-statistical-analysis: Statistical tools for zircon detection in X-ray elemental maps.*  
Zenodo. https://doi.org/10.5281/zenodo.17690656

---

## License

This project is distributed under the **MIT License**.  
See `LICENSE` for details.

---

## Contact

**Aitor Cambeses**  
Department of Mineralogy and Petrology  
University of Granada, Spain  
Faculty of Sciences, Av. de la Fuente Nueva S/N  
18071 Granada  
Tel: +34 958 243358  
Email: **aitorc@ugr.es**

---

## 📚 How to cite

If you use **ZrMaps-statistical-analysis**, please cite it as follows:

```bibtex
@software{cambeses_2025_zrmaps,
  author       = {Cambeses, Aitor},
  title        = {ZrMaps-statistical-analysis: Toolkit for the statistical detection of zircon in large X-ray elemental maps},
  year         = {2025},
  version      = {0.1.0},
  doi          = {10.5281/zenodo.17690656},
  url          = {https://github.com/aitorcambeses-ugr/ZrMaps-statistical-analysis},
  note         = {Software package archived on Zenodo. Funded by PID2023.149105NA.I00 (MCIN/AEI, Spain).}
}


TY  - COMP
T1  - ZrMaps-statistical-analysis: Toolkit for the statistical detection of zircon in large X-ray elemental maps
AU  - Cambeses, Aitor
PY  - 2025
VL  - 0.1.0
DO  - 10.5281/zenodo.17690656
UR  - https://github.com/aitorcambeses-ugr/ZrMaps-statistical-analysis
N1  - Software package archived on Zenodo. Funded by PID2023.149105NA.I00 (MCIN/AEI, Spain).
ER  - 
