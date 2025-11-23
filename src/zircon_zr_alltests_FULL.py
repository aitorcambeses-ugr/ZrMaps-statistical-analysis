#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zircon_zr_alltests_FULL.py — versión con panel PDF multipágina

Detección robusta de circones mediante 6 métodos estadísticos aplicados
a mapas de intensidad de Zr (lineal y log10). El script:

    • Ejecuta automáticamente los métodos:
        1. IQR
        2. MAD robusto
        3. Z-score clásico
        4. GMM
        5. Percentil adaptativo
        6. MAD LOCAL

    • Para cada método genera:
        - mapa PNG con leyenda incrustada
        - histograma con umbrales
        - TSV + CSV
        - TXT descriptivo del método

    • Además crea:
        - Panel combinado Zr (PNG)
        - Panel combinado logZr (PNG)
        - **UN PDF multipágina (Zr y logZr) con:**
              → imagen Zr bruta
              → 6 métodos lineales
              → imagen logZr bruta
              → 6 métodos logarítmicos
              → texto comparativo

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as patches
from sklearn.mixture import GaussianMixture

###############################################################
# 2. Utilidades: localizar y cargar mapa de Zr
###############################################################

def find_zr_file(folder):
    """
    Busca un archivo Zr.txt / Zr.tsv o *_Zr.txt/tsv dentro de folder.
    """
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            continue

        name, ext = os.path.splitext(fname)
        if ext.lower() not in (".txt", ".tsv"):
            continue

        if name == "Zr" or name.endswith("_Zr"):
            return path

    return None


def load_zr_map(folder):
    """
    Carga la matriz numérica del mapa Zr.
    """
    f = find_zr_file(folder)
    if f is None:
        raise FileNotFoundError(
            f"No se encontró Zr.txt ni *_Zr.txt en {folder}"
        )
    arr = np.loadtxt(f)
    return arr.astype(float)


###############################################################
# 3. Paleta unificada (para todos los métodos)
###############################################################

COLORS = [
    "#4b4b4b",  # 0 no zircon
    "#4aa8ff",  # 1 dubious zircon
    "#3cb371",  # 2 possible zircon
    "#ffa500",  # 3 highly probable zircon
    "#ff0000"   # 4 zircon
]

LABELS = [
    "no zircon",
    "dubious zircon",
    "possible zircon",
    "highly probable zircon",
    "zircon",
]


###############################################################
# 4. Guardar mapa PNG con leyenda incrustada
###############################################################

def save_map_with_legend(classes_img, out_png):
    """
    Guarda el mapa categórico + una leyenda incrustada tipo FRAC (opción D).
    """
    cmap = ListedColormap(COLORS)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(classes_img, cmap=cmap, norm=norm)
    ax.axis("off")

    # Crear legend a la derecha
    patches_ = [
        patches.Patch(color=c, label=l)
        for c, l in zip(COLORS, LABELS)
    ]

    plt.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    fig.legend(
        handles=patches_,
        loc="center right",
        borderaxespad=0.5,
        frameon=False
    )

    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()


###############################################################
# 5. Guardar histograma con líneas de umbral
###############################################################

def save_histogram(values, thresholds, colors, names, xlabel, out_png):
    """
    Genera un histograma con líneas verticales marcando los umbrales.

    values: array 1D con datos válidos (Zr o logZr)
    thresholds: lista [t1, t2, t3, t4]
    colors: colores para cada línea
    names: etiquetas (P75, P90, z=2, etc.)
    xlabel: etiqueta eje X
    """
    plt.figure(figsize=(7, 5))
    plt.hist(values, bins=100, color="gray", alpha=0.7)

    for t, c, n in zip(thresholds, colors, names):
        plt.axvline(t, color=c, linestyle="--", label=f"{n} = {t:.4g}")

    plt.xlabel(xlabel)
    plt.ylabel("Número de píxeles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

###############################################################
# 6. Guardar TSV (matriz de clases 0–4)
###############################################################

def save_tsv(data, out_tsv):
    """
    Guarda una matriz de clases (0–4) en formato TSV.
    """
    np.savetxt(out_tsv, data.astype(int), fmt="%d", delimiter="\t")


###############################################################
# 7. Guardar estadísticas en CSV
###############################################################

def save_stats_csv(stats, thresholds, labels, value_label, out_csv):
    """
    stats: lista de diccionarios con estadísticas por clase
    thresholds: dict con umbrales (P75, P90, etc.)
    labels: etiquetas ("no zircon", etc.)
    value_label: Zr, logZr, Zr-MAD, etc.
    """
    lines = []
    lines.append("class_id,label,n_pixels,frac_pixels,min,max")

    # estadísticas por clase
    for s in stats:
        lines.append(
            f"{s['class_id']},{labels[s['class_id']]},"
            f"{s['n_pixels']},{s['frac_pixels']:.6f},"
            f"{s['v_min']:.6g},{s['v_max']:.6g}"
        )

    # parte de umbrales
    if thresholds:
        lines.append("")  # línea en blanco
        keys = list(thresholds.keys())
        lines.append("threshold," + ",".join(keys))
        lines.append(
            value_label + "," +
            ",".join(str(thresholds[k]) for k in keys)
        )

    with open(out_csv, "w") as f:
        f.write("\n".join(lines))


###############################################################
# 8. Guardar resumen TXT (explicación del método)
###############################################################

def save_stats_txt(stats, thresholds, method_description, labels, out_txt):
    """
    Guarda un informe en texto con:
        - descripción de las clases
        - estadística por clase
        - umbrales
        - explicación del método
    """
    lines = []
    lines.append("Clasificación en 5 categorías:")
    lines.append("  0 = no zircon")
    lines.append("  1 = dubious zircon")
    lines.append("  2 = possible zircon")
    lines.append("  3 = highly probable zircon")
    lines.append("  4 = zircon\n")

    # Estadísticas por clase
    lines.append("Estadística por clase:\n")
    for s in stats:
        lines.append(f"Clase {s['class_id']} ({labels[s['class_id']]}):")
        lines.append(f"  píxeles = {s['n_pixels']}")
        lines.append(f"  fracción = {s['frac_pixels']*100:.3f} %")
        lines.append(f"  min = {s['v_min']:.6g}")
        lines.append(f"  max = {s['v_max']:.6g}\n")

    # Umbrales
    if thresholds:
        lines.append("Umbrales utilizados:")
        for k, v in thresholds.items():
            lines.append(f"  {k} = {v}")
        lines.append("")

    # Descripción del método
    lines.append("Descripción del método:\n")
    lines.append(method_description.strip())
    lines.append("")

    with open(out_txt, "w") as f:
        f.write("\n".join(lines))


###############################################################
# 9. Función genérica de clasificación con 4 umbrales
###############################################################

def classify_with_thresholds(values, t1, t2, t3, t4):
    """
    Clasificación en 5 clases:
        - ≤ t1 → 0 (no zircon)
        - t1–t2 → 1
        - t2–t3 → 2
        - t3–t4 → 3
        - > t4 → 4 (zircon)
    """
    classes = np.zeros_like(values, dtype=int)

    classes[(values > t1) & (values <= t2)] = 1
    classes[(values > t2) & (values <= t3)] = 2
    classes[(values > t3) & (values <= t4)] = 3
    classes[values > t4] = 4

    return classes
###############################################################
# 10. MÉTODO 1 — IQR (Interquartile Range)
#     Percentiles: P75 (Q3), P90, P95, P99
###############################################################

def run_IQR_method(zr, logzr, sample_dir, sample):
    print("   → Método IQR...")

    # Carpeta del método
    outdir = os.path.join(sample_dir, f"{sample}_Zr_IQR_anomaly")
    os.makedirs(outdir, exist_ok=True)

    # ==========================================================
    # A) VERSIÓN LINEAL (Zr)
    # ==========================================================
    flat = zr.flatten()
    flat_valid = flat[np.isfinite(flat)]

    # Cálculo de percentiles
    P75, P90, P95, P99 = np.percentile(flat_valid, [75, 90, 95, 99])

    # Clasificación
    classes_lin = classify_with_thresholds(zr, P75, P90, P95, P99)

    # Estadística por clase
    stats_lin = []
    tot = zr.size
    for cid in range(5):
        mask = (classes_lin == cid)
        vals = zr[mask]
        if vals.size > 0:
            vmin, vmax = float(vals.min()), float(vals.max())
        else:
            vmin = vmax = 0.0
        stats_lin.append(dict(
            class_id=cid,
            n_pixels=int(mask.sum()),
            frac_pixels=mask.sum() / tot,
            v_min=vmin,
            v_max=vmax
        ))

    thresholds_lin = {"P75": P75, "P90": P90, "P95": P95, "P99": P99}

    # Guardar resultados LINEAL
    save_tsv(classes_lin, os.path.join(outdir, f"{sample}_Zr_classes.tsv"))
    save_map_with_legend(classes_lin, os.path.join(outdir, f"{sample}_Zr_classes.png"))

    save_histogram(
        flat_valid,
        [P75, P90, P95, P99],
        ["purple", "green", "orange", "red"],
        ["P75", "P90", "P95", "P99"],
        xlabel="Intensidad Zr",
        out_png=os.path.join(outdir, f"{sample}_Zr_hist.png")
    )

    save_stats_csv(
        stats_lin, thresholds_lin, LABELS, "Zr",
        os.path.join(outdir, f"{sample}_Zr_classes.csv")
    )

    save_stats_txt(
        stats_lin, thresholds_lin,
        method_description = """
MÉTODO IQR (Interquartile Range) APLICADO A Zr (ESCALA LINEAL)

1) Procedimiento estadístico
   - Se toman todos los valores válidos de Zr del mapa.
   - Se calculan los percentiles P75, P90, P95 y P99.
   - Estos percentiles dividen la distribución en un fondo (≤P75),
     anomalías moderadas (P75–P90), anomalías claras (P90–P95) y
     anomalías extremas (P95–P99 y >P99).

2) Interpretación
   - Clases 0 y 1 representan el fondo y zonas poco enriquecidas.
   - Clases 2 y 3 marcan anomalías elevadas.
   - Clase 4 identifica los píxeles más extremos: candidatos robustos a circón.

3) Uso práctico
   - Es el método menos sesgado y más "universal".
   - Muy útil como referencia para contrastar con los demás métodos.
        """,
        labels=LABELS,
        out_txt=os.path.join(outdir, f"{sample}_Zr_stats.txt")
    )

    # ==========================================================
    # B) VERSIÓN LOGARÍTMICA (log10(Zr))
    # ==========================================================
    flatL = logzr.flatten()
    flatL_valid = flatL[np.isfinite(flatL)]

    P75L, P90L, P95L, P99L = np.percentile(flatL_valid, [75, 90, 95, 99])

    classes_log = classify_with_thresholds(logzr, P75L, P90L, P95L, P99L)

    stats_log = []
    tot = logzr.size
    for cid in range(5):
        mask = (classes_log == cid)
        vals = logzr[mask]
        if vals.size > 0:
            vmin, vmax = float(vals.min()), float(vals.max())
        else:
            vmin = vmax = 0.0
        stats_log.append(dict(
            class_id=cid,
            n_pixels=int(mask.sum()),
            frac_pixels=mask.sum() / tot,
            v_min=vmin,
            v_max=vmax
        ))

    thresholds_log = {"P75": P75L, "P90": P90L, "P95": P95L, "P99": P99L}

    save_tsv(classes_log, os.path.join(outdir, f"{sample}_logZr_classes.tsv"))
    save_map_with_legend(classes_log, os.path.join(outdir, f"{sample}_logZr_classes.png"))

    save_histogram(
        flatL_valid,
        [P75L, P90L, P95L, P99L],
        ["purple", "green", "orange", "red"],
        ["P75", "P90", "P95", "P99"],
        xlabel="log10(Zr)",
        out_png=os.path.join(outdir, f"{sample}_logZr_hist.png")
    )

    save_stats_csv(
        stats_log, thresholds_log, LABELS, "logZr",
        os.path.join(outdir, f"{sample}_logZr_classes.csv")
    )

    save_stats_txt(
        stats_log,
        thresholds_log,
        method_description = """
MÉTODO IQR APLICADO A log10(Zr)

1) Procedimiento
   - Se aplica log10(Zr) para reducir la asimetría de la distribución.
   - El cálculo de percentiles es más estable y menos sensible a valores extremos.

2) Interpretación
   - La clasificación 0–4 muestra anomalías coherentes pero con mejor separación.
   - Píxeles clasificados como 3–4 tanto aquí como en Zr lineal son
     candidatos extremadamente robustos a circones reales.

3) Uso recomendado
   - Muy útil en mapas donde existen órdenes de magnitud de diferencia en Zr.
        """,
        labels=LABELS,
        out_txt=os.path.join(outdir, f"{sample}_logZr_stats.txt")
    )
###############################################################
# 11. MÉTODO 2 — MAD robusto
#     Umbrales z = [2, 3, 4, 5]
#     x_thr = mediana + z * MAD
###############################################################

def compute_MAD_thresholds(values):
    """Devuelve mediana, MAD y umbrales absolutos med + z*MAD para z=[2,3,4,5]."""
    median = np.median(values)
    mad = np.median(np.abs(values - median)) * 1.4826  # factor de normalidad
    if mad == 0:
        mad = 1e-12  # evita división por cero para mapas muy homogéneos

    T2 = median + 2 * mad
    T3 = median + 3 * mad
    T4 = median + 4 * mad
    T5 = median + 5 * mad

    return median, mad, (T2, T3, T4, T5)


def run_MAD_method(zr, logzr, sample_dir, sample):
    print("   → Método MAD...")

    outdir = os.path.join(sample_dir, f"{sample}_Zr_MAD_anomaly")
    os.makedirs(outdir, exist_ok=True)

    # ==========================================================
    # A) VERSIÓN LINEAL
    # ==========================================================
    flat = zr.flatten()
    flat_valid = flat[np.isfinite(flat)]

    median, mad, (T2, T3, T4, T5) = compute_MAD_thresholds(flat_valid)

    classes_lin = classify_with_thresholds(zr, T2, T3, T4, T5)

    stats_lin = []
    tot = zr.size
    for cid in range(5):
        mask = (classes_lin == cid)
        vals = zr[mask]
        if vals.size > 0:
            vmin, vmax = vals.min(), vals.max()
        else:
            vmin = vmax = 0.0
        stats_lin.append(dict(
            class_id = cid,
            n_pixels = int(mask.sum()),
            frac_pixels = mask.sum() / tot,
            v_min = float(vmin),
            v_max = float(vmax)
        ))

    thresholds_lin = {
        "median": median,
        "mad": mad,
        "T2": T2, "T3": T3, "T4": T4, "T5": T5
    }

    # --- salidas versión LINEAL ---
    save_tsv(classes_lin, os.path.join(outdir, f"{sample}_Zr_classes.tsv"))
    save_map_with_legend(classes_lin, os.path.join(outdir, f"{sample}_Zr_classes.png"))

    save_histogram(
        flat_valid,
        [T2, T3, T4, T5],
        ["purple", "green", "orange", "red"],
        ["z=2", "z=3", "z=4", "z=5"],
        xlabel="Intensidad Zr",
        out_png=os.path.join(outdir, f"{sample}_Zr_hist.png")
    )

    save_stats_csv(
        stats_lin, thresholds_lin, LABELS, "Zr-MAD",
        os.path.join(outdir, f"{sample}_Zr_classes.csv")
    )

    save_stats_txt(
        stats_lin, thresholds_lin,
        method_description = """
MÉTODO MAD ROBUSTO APLICADO A Zr (ESCALA LINEAL)

1) Procedimiento estadístico
   - Se calcula la mediana de Zr en toda la imagen.
   - Se calcula el MAD (Median Absolute Deviation), que es robusto a valores anómalos.
   - Se convierten los niveles z=[2,3,4,5] a umbrales absolutos:
         T2 = med + 2*MAD
         T3 = med + 3*MAD
         T4 = med + 4*MAD
         T5 = med + 5*MAD

2) Interpretación
   - Identifica anomalías en función de su distancia robusta a la mediana.
   - Las clases 3 y 4 representan valores excepcionalmente altos.
   - Es muy fiable cuando existen grandes colas en la distribución de Zr.

3) Comparación con IQR
   - MAD es menos sensible a la forma de la distribución.
   - Suele detectar dominios de Zr extremo de manera más estable.
        """,
        labels = LABELS,
        out_txt = os.path.join(outdir, f"{sample}_Zr_stats.txt")
    )

    # ==========================================================
    # B) VERSIÓN LOGARÍTMICA
    # ==========================================================
    flatL = logzr.flatten()
    flatL_valid = flatL[np.isfinite(flatL)]

    medianL, madL, (T2L, T3L, T4L, T5L) = compute_MAD_thresholds(flatL_valid)

    classes_log = classify_with_thresholds(logzr, T2L, T3L, T4L, T5L)

    stats_log = []
    tot = logzr.size
    for cid in range(5):
        mask = (classes_log == cid)
        vals = logzr[mask]
        if vals.size > 0:
            vmin, vmax = vals.min(), vals.max()
        else:
            vmin = vmax = 0.0
        stats_log.append(dict(
            class_id = cid,
            n_pixels = int(mask.sum()),
            frac_pixels = mask.sum() / tot,
            v_min = float(vmin),
            v_max = float(vmax)
        ))

    thresholds_log = {
        "median": medianL,
        "mad": madL,
        "T2": T2L, "T3": T3L, "T4": T4L, "T5": T5L
    }

    # --- salidas LOG ---
    save_tsv(classes_log, os.path.join(outdir, f"{sample}_logZr_classes.tsv"))
    save_map_with_legend(classes_log, os.path.join(outdir, f"{sample}_logZr_classes.png"))

    save_histogram(
        flatL_valid,
        [T2L, T3L, T4L, T5L],
        ["purple", "green", "orange", "red"],
        ["z=2", "z=3", "z=4", "z=5"],
        xlabel="log10(Zr)",
        out_png=os.path.join(outdir, f"{sample}_logZr_hist.png")
    )

    save_stats_csv(
        stats_log, thresholds_log, LABELS, "logZr-MAD",
        os.path.join(outdir, f"{sample}_logZr_classes.csv")
    )

    save_stats_txt(
        stats_log, thresholds_log,
        method_description = """
MÉTODO MAD ROBUSTO APLICADO A log10(Zr)

1) Procedimiento
   - Se calcula la mediana y MAD de log10(Zr), que suele tener una distribución
     más simétrica.
   - Los umbrales se definen como med + z*MAD (z=2,3,4,5).

2) Interpretación
   - Clases 3 y 4 representan anomalías muy fuertes incluso tras comprimir la
     distribución con log10.
   - Es un método extremadamente estable para discriminar circones en mapas
     con valores muy contrastados.

3) Conclusión práctica
   - Píxeles altos en ambos (Zr y logZr) son candidatos muy robustos.
        """,
        labels = LABELS,
        out_txt = os.path.join(outdir, f"{sample}_logZr_stats.txt")
    )
###############################################################
# 12. MÉTODO 3 — Z-score clásico
#
#    z = (x - μ) / σ
#
#    Umbrales:
#       z ≤ 2  → clase 0 (no zircon)
#       2–3    → clase 1
#       3–4    → clase 2
#       4–5    → clase 3
#       >5     → clase 4 (zircon)
###############################################################

def compute_Zscore_thresholds(values):
    """Devuelve media, sigma y umbrales μ + {2σ,3σ,4σ,5σ}."""
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std == 0:
        std = 1e-12

    T2 = mean + 2 * std
    T3 = mean + 3 * std
    T4 = mean + 4 * std
    T5 = mean + 5 * std

    return mean, std, (T2, T3, T4, T5)


def run_Zscore_method(zr, logzr, sample_dir, sample):
    print("   → Método Z-score clásico...")

    outdir = os.path.join(sample_dir, f"{sample}_Zr_Zscore_anomaly")
    os.makedirs(outdir, exist_ok=True)

    # ==========================================================
    # A) Versión LINEAL
    # ==========================================================
    flat = zr.flatten()
    flat_valid = flat[np.isfinite(flat)]

    mean, std, (T2, T3, T4, T5) = compute_Zscore_thresholds(flat_valid)

    classes_lin = classify_with_thresholds(zr, T2, T3, T4, T5)

    stats_lin = []
    tot = zr.size
    for cid in range(5):
        mask = (classes_lin == cid)
        vals = zr[mask]
        if vals.size > 0:
            vmin, vmax = vals.min(), vals.max()
        else:
            vmin = vmax = 0.0

        stats_lin.append(dict(
            class_id   = cid,
            n_pixels   = int(mask.sum()),
            frac_pixels = mask.sum() / tot,
            v_min = float(vmin),
            v_max = float(vmax)
        ))

    thresholds_lin = {
        "mean": mean,
        "std": std,
        "T2": T2, "T3": T3, "T4": T4, "T5": T5
    }

    # --- salidas LINEAL ---
    save_tsv(classes_lin, os.path.join(outdir, f"{sample}_Zr_classes.tsv"))
    save_map_with_legend(classes_lin, os.path.join(outdir, f"{sample}_Zr_classes.png"))

    save_histogram(
        flat_valid,
        [T2, T3, T4, T5],
        ["purple","green","orange","red"],
        ["z=2","z=3","z=4","z=5"],
        xlabel="Intensidad Zr",
        out_png=os.path.join(outdir, f"{sample}_Zr_hist.png")
    )

    save_stats_csv(
        stats_lin, thresholds_lin, LABELS, "Zr-Zscore",
        os.path.join(outdir, f"{sample}_Zr_classes.csv")
    )

    save_stats_txt(
        stats_lin, thresholds_lin,
        method_description = """
MÉTODO Z-SCORE CLÁSICO APLICADO A Zr (ESCALA LINEAL)

1) Procedimiento estadístico
   - Se calcula la media μ y la desviación estándar σ de todos los valores Zr.
   - Se generan umbrales absolutos:
        T2 = μ + 2σ
        T3 = μ + 3σ
        T4 = μ + 4σ
        T5 = μ + 5σ

2) Interpretación
   - Clase 0: fondo estadístico de Zr.
   - Clases 1–2: enriquecimiento progresivo respecto a μ.
   - Clases 3–4: valores muy raros desde una distribución gaussiana,
     candidatos fuertes a circones.

3) Limitaciones
   - Si la distribución está muy sesgada, μ y σ pueden quedar distorsionadas.
   - Por eso es importante comparar con MAD (robusto).
        """,
        labels=LABELS,
        out_txt=os.path.join(outdir, f"{sample}_Zr_stats.txt")
    )

    # ==========================================================
    # B) Versión LOGARÍTMICA
    # ==========================================================
    flatL = logzr.flatten()
    flatL_valid = flatL[np.isfinite(flatL)]

    meanL, stdL, (T2L, T3L, T4L, T5L) = compute_Zscore_thresholds(flatL_valid)

    classes_log = classify_with_thresholds(logzr, T2L, T3L, T4L, T5L)

    stats_log = []
    tot = logzr.size
    for cid in range(5):
        mask = (classes_log == cid)
        vals = logzr[mask]
        if vals.size > 0:
            vmin, vmax = vals.min(), vals.max()
        else:
            vmin = vmax = 0.0

        stats_log.append(dict(
            class_id = cid,
            n_pixels = int(mask.sum()),
            frac_pixels = mask.sum() / tot,
            v_min = float(vmin),
            v_max = float(vmax)
        ))

    thresholds_log = {
        "mean": meanL,
        "std":  stdL,
        "T2": T2L, "T3": T3L, "T4": T4L, "T5": T5L
    }

    # --- salidas LOG ---
    save_tsv(classes_log, os.path.join(outdir, f"{sample}_logZr_classes.tsv"))
    save_map_with_legend(classes_log, os.path.join(outdir, f"{sample}_logZr_classes.png"))

    save_histogram(
        flatL_valid,
        [T2L, T3L, T4L, T5L],
        ["purple","green","orange","red"],
        ["z=2","z=3","z=4","z=5"],
        xlabel="log10(Zr)",
        out_png=os.path.join(outdir, f"{sample}_logZr_hist.png")
    )

    save_stats_csv(
        stats_log, thresholds_log, LABELS, "logZr-Zscore",
        os.path.join(outdir, f"{sample}_logZr_classes.csv")
    )

    save_stats_txt(
        stats_log, thresholds_log,
        method_description = """
MÉTODO Z-SCORE APLICADO A log10(Zr)

1) Por qué usar log10(Zr)
   - La escala logarítmica suele producir distribuciones más cercanas a una
     gaussiana → el Z-score es más fiable.

2) Interpretación
   - Clases altas (3–4) corresponden a dominios que permanecen anómalos
     incluso después de transformar la distribución en log.
   - Muy útil para detectar circones en mapas donde hay valores muy extremos.

3) Recomendación
   - Comparar las clases altas de Zr y logZr permite identificar anomalías
     verdaderamente robustas.
        """,
        labels=LABELS,
        out_txt=os.path.join(outdir, f"{sample}_logZr_stats.txt")
    )
###############################################################
# 13. MÉTODO 4 — Gaussian Mixture Model (GMM)
#
#   Encuentra dos poblaciones:
#       componente 0 → fondo
#       componente 1 → alta Zr
#
#   Probabilidad p = P(componente alta | intensidad)
#
#   Umbrales:
#       p < 0.50     → Clase 0
#       0.50–0.80    → Clase 1
#       0.80–0.95    → Clase 2
#       0.95–0.99    → Clase 3
#       >0.99        → Clase 4 (zircon)
###############################################################

def gmm_thresholds(probs):
    """Devuelve los cuatro umbrales de probabilidad."""
    return (0.50, 0.80, 0.95, 0.99)


def run_GMM_method(zr, logzr, sample_dir, sample):
    print("   → Método GMM...")

    outdir = os.path.join(sample_dir, f"{sample}_Zr_GMM_anomaly")
    os.makedirs(outdir, exist_ok=True)

    # ==========================================================
    # A) LINEAL
    # ==========================================================

    flat = zr.flatten()
    flat_valid = flat[np.isfinite(flat)].reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
    gmm.fit(flat_valid)

    probs = gmm.predict_proba(flat_valid)[:, 1]   # prob de componente rica en Zr
    probs_img = probs.reshape(zr.shape)

    # Umbrales
    T1, T2, T3, T4 = gmm_thresholds(probs)
    classes_lin = classify_with_thresholds(probs_img, T1, T2, T3, T4)

    # Estadísticas
    stats_lin = []
    tot = zr.size
    for cid in range(5):
        mask = (classes_lin == cid)
        vals = flat[mask.flatten()]
        if vals.size > 0:
            vmin, vmax = vals.min(), vals.max()
        else:
            vmin = vmax = 0.0

        stats_lin.append(dict(
            class_id    = cid,
            n_pixels    = int(mask.sum()),
            frac_pixels = mask.sum() / tot,
            v_min       = float(vmin),
            v_max       = float(vmax)
        ))

    thresholds_lin = {
        "p50": T1,
        "p80": T2,
        "p95": T3,
        "p99": T4
    }

    # Salidas LINEALES
    save_tsv(classes_lin, os.path.join(outdir, f"{sample}_Zr_classes.tsv"))
    save_map_with_legend(classes_lin, os.path.join(outdir, f"{sample}_Zr_classes.png"))

    save_histogram(
        probs,
        [T1, T2, T3, T4],
        ["purple", "green", "orange", "red"],
        ["p50", "p80", "p95", "p99"],
        xlabel="P(componente alta)",
        out_png=os.path.join(outdir, f"{sample}_Zr_hist.png"),
    )

    save_stats_csv(stats_lin, thresholds_lin, LABELS, "Zr-GMM",
                   os.path.join(outdir, f"{sample}_Zr_classes.csv"))

    save_stats_txt(
        stats_lin,
        thresholds_lin,
        method_description = """
MÉTODO GMM (Gaussian Mixture Model) APLICADO A Zr

1) Procedimiento
   - Se ajusta una mezcla de 2 gaussianas a la distribución de intensidades Zr.
   - De este ajuste se obtiene para cada píxel una probabilidad p de que
     pertenezca a la población "Zr alto".

2) Clasificación basada en probabilidad
       Clase 0: p < 0.50  (fondo)
       Clase 1: 0.50–0.80
       Clase 2: 0.80–0.95
       Clase 3: 0.95–0.99
       Clase 4: p ≥ 0.99  (zircon)

3) Interpretación
   - Clases 3–4 son píxeles que casi con total certeza pertenecen a la población
     enriquecida en Zr → candidatos robustos a circones.

4) Ventajas
   - Muy eficaz cuando la distribución presenta dos poblaciones claras.
   - No depende de umbrales de intensidad absoluta.
        """,
        labels=LABELS,
        out_txt=os.path.join(outdir, f"{sample}_Zr_stats.txt"),
    )

    # ==========================================================
    # B) LOGARÍTMICO
    # ==========================================================

    flatL = logzr.flatten()
    flatL_valid = flatL[np.isfinite(flatL)].reshape(-1, 1)

    gmmL = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
    gmmL.fit(flatL_valid)
    probsL = gmmL.predict_proba(flatL_valid)[:, 1]

    probsL_img = probsL.reshape(logzr.shape)

    T1L, T2L, T3L, T4L = gmm_thresholds(probsL)
    classes_log = classify_with_thresholds(probsL_img, T1L, T2L, T3L, T4L)

    stats_log = []
    tot = logzr.size
    for cid in range(5):
        mask = (classes_log == cid)
        vals = flatL[mask.flatten()]
        if vals.size > 0:
            vmin, vmax = vals.min(), vals.max()
        else:
            vmin = vmax = 0.0

        stats_log.append(dict(
            class_id    = cid,
            n_pixels    = int(mask.sum()),
            frac_pixels = mask.sum() / tot,
            v_min       = float(vmin),
            v_max       = float(vmax)
        ))

    thresholds_log = {
        "p50": T1L,
        "p80": T2L,
        "p95": T3L,
        "p99": T4L
    }

    # Salidas LOG
    save_tsv(classes_log, os.path.join(outdir, f"{sample}_logZr_classes.tsv"))
    save_map_with_legend(classes_log, os.path.join(outdir, f"{sample}_logZr_classes.png"))

    save_histogram(
        probsL,
        [T1L, T2L, T3L, T4L],
        ["purple", "green", "orange", "red"],
        ["p50", "p80", "p95", "p99"],
        xlabel="P(componente alta) en log10(Zr)",
        out_png=os.path.join(outdir, f"{sample}_logZr_hist.png"),
    )

    save_stats_csv(
        stats_log, thresholds_log, LABELS, "logZr-GMM",
        os.path.join(outdir, f"{sample}_logZr_classes.csv"),
    )

    save_stats_txt(
        stats_log,
        thresholds_log,
        method_description = """
MÉTODO GMM APLICADO A log10(Zr)

- El ajuste en log10(Zr) suele separar aún mejor las dos gaussianas.
- La probabilidad p refleja ahora la mezcla en el espacio transformado.
- Píxeles que pertenecen a clases altas en ambos espacios (lineal y log)
  representan anomalías extremadamente robustas.
        """,
        labels=LABELS,
        out_txt=os.path.join(outdir, f"{sample}_logZr_stats.txt"),
    )

###############################################################
# 14. MÉTODO 5 — Percentil Adaptativo
#
#   Percentiles:
#       P85, P90, P95, P99.5
#
#   Clasificación:
#       Z ≤ P85       → Clase 0 (no zircon)
#       P85–P90       → Clase 1
#       P90–P95       → Clase 2
#       P95–P99.5     → Clase 3
#       > P99.5       → Clase 4 (zircon)
###############################################################

def run_Adaptive_method(zr, logzr, sample_dir, sample):
    print("   → Método Adaptativo (P85–P90–P95–P99.5)...")

    outdir = os.path.join(sample_dir, f"{sample}_Zr_Adaptive_anomaly")
    os.makedirs(outdir, exist_ok=True)

    # ==========================================================
    # A) LINEAL
    # ==========================================================

    flat = zr.flatten()
    flat_valid = flat[np.isfinite(flat)]

    # percentiles adaptativos
    P85, P90, P95, P995 = np.percentile(flat_valid, [85, 90, 95, 99.5])

    classes_lin = classify_with_thresholds(zr, P85, P90, P95, P995)

    # Estadísticas
    stats_lin = []
    tot = zr.size
    for cid in range(5):
        mask = (classes_lin == cid)
        vals = zr[mask]

        if vals.size > 0:
            vmin, vmax = float(vals.min()), float(vals.max())
        else:
            vmin = vmax = 0.0

        stats_lin.append(dict(
            class_id     = cid,
            n_pixels     = int(mask.sum()),
            frac_pixels  = mask.sum() / tot,
            v_min        = vmin,
            v_max        = vmax
        ))

    thresholds_lin = {
        "P85": P85,
        "P90": P90,
        "P95": P95,
        "P99.5": P995
    }

    # Salidas
    save_tsv(classes_lin, os.path.join(outdir, f"{sample}_Zr_classes.tsv"))
    save_map_with_legend(classes_lin, os.path.join(outdir, f"{sample}_Zr_classes.png"))

    save_histogram(
        flat_valid,
        [P85, P90, P95, P995],
        ["purple", "green", "orange", "red"],
        ["P85", "P90", "P95", "P99.5"],
        xlabel="Intensidad Zr",
        out_png=os.path.join(outdir, f"{sample}_Zr_hist.png"),
    )

    save_stats_csv(
        stats_lin, thresholds_lin, LABELS, "Zr-Adapt",
        os.path.join(outdir, f"{sample}_Zr_classes.csv")
    )

    save_stats_txt(
        stats_lin,
        thresholds_lin,
        method_description = """
MÉTODO DE PERCENTIL ADAPTATIVO (P85–P90–P95–P99.5)

1) Procedimiento
   - Se calculan percentiles avanzados que eliminan casi todo el fondo
     y dejan únicamente la parte alta de la distribución.
   - Es más estricto que IQR y resalta únicamente las anomalías fuertes.

2) Interpretación
       · Clase 0 (Zr ≤ P85): fondo estadístico
       · Clase 1 (P85–P90): anomalía débil
       · Clase 2 (P90–P95): anomalía moderada
       · Clase 3 (P95–P99.5): anomalía fuerte
       · Clase 4 (>P99.5): extremo → circones o dominios muy Zr-rico

3) Ventajas
   - Minimiza falsos positivos.
   - Ideal como filtro previo para localizar objetivos prioritarios.
        """,
        labels=LABELS,
        out_txt=os.path.join(outdir, f"{sample}_Zr_stats.txt")
    )


    # ==========================================================
    # B) LOGARÍTMICO
    # ==========================================================

    flatL = logzr.flatten()
    flatL_valid = flatL[np.isfinite(flatL)]

    P85L, P90L, P95L, P995L = np.percentile(flatL_valid, [85, 90, 95, 99.5])

    classes_log = classify_with_thresholds(logzr, P85L, P90L, P95L, P995L)

    stats_log = []
    tot = logzr.size
    for cid in range(5):
        mask = (classes_log == cid)
        vals = logzr[mask]

        if vals.size > 0:
            vmin, vmax = float(vals.min()), float(vals.max())
        else:
            vmin = vmax = 0.0

        stats_log.append(dict(
            class_id     = cid,
            n_pixels     = int(mask.sum()),
            frac_pixels  = mask.sum() / tot,
            v_min        = vmin,
            v_max        = vmax
        ))

    thresholds_log = {
        "P85": P85L,
        "P90": P90L,
        "P95": P95L,
        "P99.5": P995L
    }

    # Salidas LOG
    save_tsv(classes_log, os.path.join(outdir, f"{sample}_logZr_classes.tsv"))
    save_map_with_legend(classes_log, os.path.join(outdir, f"{sample}_logZr_classes.png"))

    save_histogram(
        flatL_valid,
        [P85L, P90L, P95L, P995L],
        ["purple", "green", "orange", "red"],
        ["P85", "P90", "P95", "P99.5"],
        xlabel="log10(Zr)",
        out_png=os.path.join(outdir, f"{sample}_logZr_hist.png"),
    )

    save_stats_csv(
        stats_log, thresholds_log, LABELS, "logZr-Adapt",
        os.path.join(outdir, f"{sample}_logZr_classes.csv")
    )

    save_stats_txt(
        stats_log,
        thresholds_log,
        method_description = """
MÉTODO ADAPTATIVO APLICADO A log10(Zr)

- Usa los mismos percentiles P85–P99.5, pero aplicados sobre log10(Zr).
- Muy útil para detectar valores extremos cuando el rango dinámico es alto.
- Clasificaciones 3–4 en logZr representan anomalías extremadamente intensas.
        """,
        labels=LABELS,
        out_txt=os.path.join(outdir, f"{sample}_logZr_stats.txt")
    )

###############################################################
# 15. MÉTODO 6 — MAD LOCAL (ventana 21 / 25 / 31 px)
#
#   - Se calcula, para cada píxel, la mediana_local y MAD_local
#     en una ventana centrada en él.
#   - Tamaño de ventana (según tamaño de imagen):
#       max(Nx, Ny) <  400 → 21 x 21
#       400–1000           → 25 x 25
#       > 1000             → 31 x 31
#
#   z_local = (x - mediana_local) / (1.4826 * MAD_local)
#
#   Umbrales:
#       z ≤ 2     → clase 0
#       2–3       → clase 1
#       3–4       → clase 2
#       4–5       → clase 3
#       >5        → clase 4
#
#   Mucho más sensible a anomalías "puntuales" respecto a su matriz.
###############################################################

def auto_window_size(shape):
    """
    Elige tamaño de ventana 21 / 25 / 31 en función del tamaño de la imagen.
    """
    ny, nx = shape
    m = max(nx, ny)
    if m < 400:
        return 21
    elif m < 1000:
        return 25
    else:
        return 31


def compute_local_MAD_map(values):
    """
    Calcula mapas de mediana_local y mad_local usando una ventana
    auto-ajustada según el tamaño de la imagen.
    Implementación sencilla con doble bucle (más lenta en mapas grandes).
    """
    h, w = values.shape
    win = auto_window_size(values.shape)
    r = win // 2

    # padding para bordes
    padded = np.pad(values, pad_width=r, mode="edge")

    med_map = np.zeros_like(values, dtype=float)
    mad_map = np.zeros_like(values, dtype=float)

    for i in range(h):
        for j in range(w):
            sub = padded[i:i+win, j:j+win]
            sub_flat = sub.flatten()
            m = np.median(sub_flat)
            mad = np.median(np.abs(sub_flat - m)) * 1.4826
            if mad == 0:
                mad = 1e-12
            med_map[i, j] = m
            mad_map[i, j] = mad

    return med_map, mad_map, win


def classify_with_local_z(values, med_map, mad_map):
    """
    Calcula z_local = (values - med_map) / mad_map
    y devuelve clases 0–4 usando umbrales z=[2,3,4,5].
    """
    z = (values - med_map) / mad_map
    classes = np.zeros_like(values, dtype=int)

    classes[(z > 2) & (z <= 3)] = 1
    classes[(z > 3) & (z <= 4)] = 2
    classes[(z > 4) & (z <= 5)] = 3
    classes[z > 5] = 4

    return classes, z


def run_LocalMAD_method(zr, logzr, sample_dir, sample):
    print("   → Método MAD LOCAL (esto puede tardar)...")

    outdir = os.path.join(sample_dir, f"{sample}_Zr_LocalMAD_anomaly")
    os.makedirs(outdir, exist_ok=True)

    # ==========================================================
    # A) LINEAL
    # ==========================================================
    med_map, mad_map, win = compute_local_MAD_map(zr)
    classes_lin, z_local = classify_with_local_z(zr, med_map, mad_map)

    flat = zr.flatten()
    flat_valid = flat[np.isfinite(flat)]

    # Estadísticas por clase en términos de Zr
    stats_lin = []
    tot = zr.size
    for cid in range(5):
        mask = (classes_lin == cid)
        vals = zr[mask]
        if vals.size > 0:
            vmin, vmax = float(vals.min()), float(vals.max())
        else:
            vmin = vmax = 0.0

        stats_lin.append(dict(
            class_id     = cid,
            n_pixels     = int(mask.sum()),
            frac_pixels  = mask.sum() / tot,
            v_min        = vmin,
            v_max        = vmax
        ))

    thresholds_lin = {
        "ventana_px": win,
        "z1": 2.0,
        "z2": 3.0,
        "z3": 4.0,
        "z4": 5.0
    }

    # Guardar mapas y stats (LINEAL)
    save_tsv(classes_lin, os.path.join(outdir, f"{sample}_Zr_classes.tsv"))
    save_map_with_legend(classes_lin, os.path.join(outdir, f"{sample}_Zr_classes.png"))

    # Histograma de z_local
    z_flat = z_local.flatten()
    z_valid = z_flat[np.isfinite(z_flat)]

    save_histogram(
        z_valid,
        [2.0, 3.0, 4.0, 5.0],
        ["purple", "green", "orange", "red"],
        ["z=2", "z=3", "z=4", "z=5"],
        xlabel="z_local (MAD)",
        out_png=os.path.join(outdir, f"{sample}_Zr_hist.png")
    )

    save_stats_csv(
        stats_lin, thresholds_lin, LABELS, "Zr-LocalMAD",
        os.path.join(outdir, f"{sample}_Zr_classes.csv")
    )

    save_stats_txt(
        stats_lin, thresholds_lin,
        method_description = f"""
MÉTODO MAD LOCAL APLICADO A Zr (VENTANA {win}x{win})

1) Procedimiento
   - Para cada píxel se toma una ventana local {win}x{win}.
   - Se calcula la mediana_local y MAD_local de Zr.
   - Se define:
       z_local = (Zr - mediana_local) / MAD_local
   - Se asignan clases:
       · Clase 0: z_local ≤ 2
       · Clase 1: 2–3
       · Clase 2: 3–4
       · Clase 3: 4–5
       · Clase 4: >5

2) Qué detecta
   - Anomalías muy locales respecto al entorno inmediato.
   - Excelente para resaltar granos pequeños o aislados de circón.

3) Comparación con métodos globales
   - Mientras IQR, MAD y Z-score comparan con toda la imagen,
     MAD LOCAL compara con el vecindario.
   - La combinación de ambos enfoques (global + local) es muy potente.
        """,
        labels=LABELS,
        out_txt=os.path.join(outdir, f"{sample}_Zr_stats.txt")
    )

    # ==========================================================
    # B) LOGARÍTMICO
    # ==========================================================

    med_mapL, mad_mapL, winL = compute_local_MAD_map(logzr)
    classes_log, z_localL = classify_with_local_z(logzr, med_mapL, mad_mapL)

    flatL = logzr.flatten()
    flatL_valid = flatL[np.isfinite(flatL)]

    stats_log = []
    tot = logzr.size
    for cid in range(5):
        mask = (classes_log == cid)
        vals = logzr[mask]

        if vals.size > 0:
            vmin, vmax = float(vals.min()), float(vals.max())
        else:
            vmin = vmax = 0.0

        stats_log.append(dict(
            class_id    = cid,
            n_pixels    = int(mask.sum()),
            frac_pixels = mask.sum() / tot,
            v_min       = vmin,
            v_max       = vmax
        ))

    thresholds_log = {
        "ventana_px": winL,
        "z1": 2.0,
        "z2": 3.0,
        "z3": 4.0,
        "z4": 5.0
    }

    save_tsv(classes_log, os.path.join(outdir, f"{sample}_logZr_classes.tsv"))
    save_map_with_legend(classes_log, os.path.join(outdir, f"{sample}_logZr_classes.png"))

    # Histograma de z_local en logZr
    z_flatL = z_localL.flatten()
    z_validL = z_flatL[np.isfinite(z_flatL)]

    save_histogram(
        z_validL,
        [2.0, 3.0, 4.0, 5.0],
        ["purple", "green", "orange", "red"],
        ["z=2", "z=3", "z=4", "z=5"],
        xlabel="z_local (MAD) sobre log10(Zr)",
        out_png=os.path.join(outdir, f"{sample}_logZr_hist.png")
    )

    save_stats_csv(
        stats_log, thresholds_log, LABELS, "logZr-LocalMAD",
        os.path.join(outdir, f"{sample}_logZr_classes.csv")
    )

    save_stats_txt(
        stats_log, thresholds_log,
        method_description = f"""
MÉTODO MAD LOCAL APLICADO A log10(Zr) (VENTANA {winL}x{winL})

1) Procedimiento
   - Igual que en Zr lineal, pero sobre log10(Zr).
   - El log reduce el rango dinámico y el MAD local realza anomalías
     relativas al entorno inmediato.

2) Ventajas
   - Detección especialmente robusta de granos de circón pequeños.
   - Muy útil en mapas donde el fondo presenta variaciones suaves.

3) Interpretación conjunta
   - Píxeles en clases 3–4 en métodos globales (IQR/MAD/GMM) y
     simultáneamente en clases altas de MAD LOCAL son candidatos
     prioritarios para validación con SEM/CL/SHRIMP.
        """,
        labels=LABELS,
        out_txt=os.path.join(outdir, f"{sample}_logZr_stats.txt")
    )

###############################################################
# 16. Panel combinado de los 6 métodos (Zr y logZr)
#
#     Orden lógico (de menos robusto → más robusto):
#       1) Zscore
#       2) IQR
#       3) MAD
#       4) Adaptive
#       5) GMM
#       6) LocalMAD
#
#   - Figuras grandes, mínimo espacio en blanco
#   - Una sola leyenda pequeña centrada
###############################################################

def combine_six_panels(sample_dir, sample):
    import matplotlib.image as mpimg
    import matplotlib.patches as patches

    # Nuevo orden lógico
    methods = ["Zscore", "IQR", "MAD", "Adaptive", "GMM", "LocalMAD"]

    # ------------------------------------------------------------------
    # PANEL Zr (lineal)
    # ------------------------------------------------------------------
    imgs_lin = []
    for m in methods:
        path = os.path.join(
            sample_dir,
            f"{sample}_Zr_{m}_anomaly",
            f"{sample}_Zr_classes.png"
        )
        imgs_lin.append(mpimg.imread(path) if os.path.isfile(path) else None)

    fig, axes = plt.subplots(2, 3, figsize=(28, 16))
    axs = axes.flatten()

    for ax, img, m in zip(axs, imgs_lin, methods):
        if img is None:
            ax.text(0.5, 0.5, f"{m} PNG missing", ha="center", va="center")
        else:
            ax.imshow(img)
            ax.set_title(f"{m} (Zr)", fontsize=20)
        ax.axis("off")

    # ---- Leyenda pequeña única ----
    colors = ["#4b4b4b","#4aa8ff","#3cb371","#ffa500","#ff0000"]
    labels = ["no zircon","dubious zircon","possible zircon",
              "highly probable zircon","zircon"]

    patches_ = [patches.Patch(color=c, label=l) for c,l in zip(colors,labels)]
    fig.legend(handles=patches_, loc="lower center",
               ncol=5, fontsize=16, frameon=False)

    plt.subplots_adjust(wspace=0.02, hspace=-0.05, bottom=0.15)
    out_lin = os.path.join(sample_dir, f"{sample}_Zr_combined_panel.png")
    plt.savefig(out_lin, dpi=300, bbox_inches="tight")
    plt.close()


    # ------------------------------------------------------------------
    # PANEL logZr
    # ------------------------------------------------------------------
    imgs_log = []
    for m in methods:
        path = os.path.join(
            sample_dir,
            f"{sample}_Zr_{m}_anomaly",
            f"{sample}_logZr_classes.png"
        )
        imgs_log.append(mpimg.imread(path) if os.path.isfile(path) else None)

    fig, axes = plt.subplots(2, 3, figsize=(28, 16))
    axs = axes.flatten()

    for ax, img, m in zip(axs, imgs_log, methods):
        if img is None:
            ax.text(0.5, 0.5, f"{m} PNG missing", ha="center", va="center")
        else:
            ax.imshow(img)
            ax.set_title(f"{m} (logZr)", fontsize=20)
        ax.axis("off")

    # ---- Leyenda pequeña única ----
    patches_ = [patches.Patch(color=c, label=l) for c,l in zip(colors,labels)]
    fig.legend(handles=patches_, loc="lower center",
               ncol=5, fontsize=16, frameon=False)

    plt.subplots_adjust(wspace=0.02, hspace=-0.05, bottom=0.15)
    out_log = os.path.join(sample_dir, f"{sample}_logZr_combined_panel.png")
    plt.savefig(out_log, dpi=300, bbox_inches="tight")
    plt.close()
###############################################################
# 17. PDF multipágina:
#
#   a) <sample>_Zr_allmethods.pdf  (2 páginas)
#      - Página 1: panel Zr (6 métodos, una sola leyenda)
#      - Página 2: panel logZr (6 métodos, una sola leyenda)
#
#   b) <sample>_Zr_maps.pdf        (1 página)
#      - Mapa Zr (lineal) + mapa log10(Zr) con barras de color
###############################################################

def build_combined_panel_with_Zr_pdf(sample_dir, sample, zr, logzr):
    import matplotlib.image as mpimg
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.patches as patches

    methods = ["IQR", "MAD", "Zscore", "GMM", "Adaptive", "LocalMAD"]

    # ------------------------------------------------------------------
    # Rutas de los PNG de clasificación (ya generados por los métodos)
    # ------------------------------------------------------------------
    panel_lin_paths = [
        os.path.join(sample_dir, f"{sample}_Zr_{m}_anomaly", f"{sample}_Zr_classes.png")
        for m in methods
    ]

    panel_log_paths = [
        os.path.join(sample_dir, f"{sample}_Zr_{m}_anomaly", f"{sample}_logZr_classes.png")
        for m in methods
    ]

    # ------------------------------------------------------------------
    # 17.a  PDF con paneles de los 6 métodos (lineal + logZr)
    # ------------------------------------------------------------------
    out_pdf_all = os.path.join(sample_dir, f"{sample}_Zr_allmethods.pdf")

    with PdfPages(out_pdf_all) as pdf:

        # ===================== PÁGINA 1 – Zr lineal ====================
        fig1, axes = plt.subplots(2, 3, figsize=(13, 8))
        axs = axes.flatten()

        for ax, img_path, m in zip(axs, panel_lin_paths, methods):
            if os.path.isfile(img_path):
                img = mpimg.imread(img_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, f"{m}: PNG missing", ha="center", va="center")

            ax.set_title(f"{m} (Zr)", fontsize=10)
            ax.axis("off")

        # Leyenda única para las 6 figuras
        legend_patches = [
            patches.Patch(color="#4b4b4b", label="no zircon"),
            patches.Patch(color="#4aa8ff", label="dubious zircon"),
            patches.Patch(color="#3cb371", label="possible zircon"),
            patches.Patch(color="#ffa500", label="highly probable zircon"),
            patches.Patch(color="#ff0000", label="zircon"),
        ]
        fig1.legend(handles=legend_patches,
                    loc="lower center",
                    ncol=5,
                    fontsize=10,
                    frameon=False,
                    bbox_to_anchor=(0.5, -0.02))

        fig1.suptitle("Clasificación Zr (6 métodos) – Escala lineal", fontsize=14)
        fig1.tight_layout(rect=[0, 0.05, 1, 0.97])
        pdf.savefig(fig1)
        plt.close(fig1)

        # ===================== PÁGINA 2 – log10(Zr) ====================
        fig2, axes = plt.subplots(2, 3, figsize=(13, 8))
        axs = axes.flatten()

        for ax, img_path, m in zip(axs, panel_log_paths, methods):
            if os.path.isfile(img_path):
                img = mpimg.imread(img_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, f"{m}: PNG missing", ha="center", va="center")

            ax.set_title(f"{m} (logZr)", fontsize=10)
            ax.axis("off")

        fig2.legend(handles=legend_patches,
                    loc="lower center",
                    ncol=5,
                    fontsize=10,
                    frameon=False,
                    bbox_to_anchor=(0.5, -0.02))

        fig2.suptitle("Clasificación Zr (6 métodos) – log10(Zr)", fontsize=14)
        fig2.tight_layout(rect=[0, 0.05, 1, 0.97])
        pdf.savefig(fig2)
        plt.close(fig2)

    print(f">> PDF de paneles generado: {out_pdf_all}")

    # ------------------------------------------------------------------
    # 17.b  PDF con mapas originales Zr y logZr
    # ------------------------------------------------------------------
    out_pdf_maps = os.path.join(sample_dir, f"{sample}_Zr_maps.pdf")

    with PdfPages(out_pdf_maps) as pdf_maps:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Zr lineal
        im0 = axes[0].imshow(zr, cmap="viridis")
        axes[0].set_title(f"{sample} – Zr (intensidad lineal)", fontsize=11)
        axes[0].axis("off")
        cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        cbar0.set_label("Intensidad Zr", fontsize=9)

        # log10(Zr)
        im1 = axes[1].imshow(logzr, cmap="viridis")
        axes[1].set_title(f"{sample} – log10(Zr)", fontsize=11)
        axes[1].axis("off")
        cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        cbar1.set_label("log10(Intensidad Zr)", fontsize=9)

        fig.suptitle("Mapas originales de Zr y log10(Zr)", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        pdf_maps.savefig(fig)
        plt.close(fig)

    print(f">> PDF de mapas Zr/logZr generado: {out_pdf_maps}")

###############################################################
# 18. MAIN
###############################################################

def main():
    if len(sys.argv) < 2:
        print("USO: python3 zircon_zr_alltests_FULL.py <nombre_muestra>")
        print("Ejemplo: python3 zircon_zr_alltests_FULL.py test")
        sys.exit(1)

    sample = sys.argv[1]
    root = os.path.abspath(os.getcwd())
    sample_dir = os.path.join(root, sample)
    mapas_dir = os.path.join(sample_dir, f"{sample}_mapas_tsv")

    if not os.path.isdir(mapas_dir):
        print("[ERROR] No existe la carpeta de mapas:", mapas_dir)
        sys.exit(1)

    print(f">> Muestra: {sample}")
    print(f">> Carpeta de mapas: {mapas_dir}")

    # 1) Cargar mapa de Zr
    zr = load_zr_map(mapas_dir)

    # 2) Construir log10(Zr) con tratamiento de ceros
    zr_pos = np.where(zr > 0, zr, np.nan)
    logzr = np.log10(zr_pos)
    if np.isfinite(logzr).any():
        mn = np.nanmin(logzr)
        logzr = np.where(np.isfinite(logzr), logzr, mn)
    else:
        logzr = np.zeros_like(zr)

    # 3) Ejecutar los 6 métodos
    print(">> Ejecutando IQR...")
    run_IQR_method(zr, logzr, sample_dir, sample)

    print(">> Ejecutando MAD global...")
    run_MAD_method(zr, logzr, sample_dir, sample)

    print(">> Ejecutando Z-score clásico...")
    run_Zscore_method(zr, logzr, sample_dir, sample)

    print(">> Ejecutando GMM...")
    run_GMM_method(zr, logzr, sample_dir, sample)

    print(">> Ejecutando Percentil Adaptativo...")
    run_Adaptive_method(zr, logzr, sample_dir, sample)

    print(">> Ejecutando MAD LOCAL...")
    run_LocalMAD_method(zr, logzr, sample_dir, sample)

    # 4) Paneles combinados en PNG
    print(">> Generando paneles combinados (Zr y logZr)...")
    combine_six_panels(sample_dir, sample)

    # 5) PDF multipágina con Zr + logZr
    print(">> Generando PDF multipágina con Zr + logZr...")
    build_combined_panel_with_Zr_pdf(sample_dir, sample, zr, logzr)

    print(">> TODO LISTO.")


if __name__ == "__main__":
    main()







