#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zircon_zr_DBSCAN_HDBSCAN.py

Detección de granos de circón mediante clustering de densidad (DBSCAN / HDBSCAN)
aplicado a mapas de Zr.

Entrada:
    python3 zircon_zr_DBSCAN_HDBSCAN.py <muestra>

    - Busca el mapa de Zr en:
          <muestra>/<muestra>_mapas_tsv/Zr.txt
      o bien en:
          <muestra>/<muestra>_mapas_tsv/*_Zr.txt

Salida (por defecto):
    <muestra>/<muestra>_Zr_DBSCAN/
        *_Zr_DBSCAN_labels.tsv
        *_Zr_DBSCAN_clusters.png
        *_Zr_DBSCAN_summary.csv
        *_Zr_DBSCAN_classes.tsv           (clases 0–4 tipo "zircon")
        *_Zr_DBSCAN_classes.png
        *_Zr_DBSCAN_stats.txt

    <muestra>/<muestra>_Zr_HDBSCAN/      (solo si hdbscan está instalado)
        *_Zr_HDBSCAN_labels.tsv
        *_Zr_HDBSCAN_probabilities.tsv
        *_Zr_HDBSCAN_clusters.png
        *_Zr_HDBSCAN_probabilities.png
        *_Zr_HDBSCAN_summary.csv
        *_Zr_HDBSCAN_classes.tsv          (clases 0–4 tipo "zircon")
        *_Zr_HDBSCAN_classes.png
        *_Zr_HDBSCAN_stats.txt

Las clases 0–4 se definen de forma análoga a los tests estadísticos:
    0 = no zircon
    1 = dubious zircon
    2 = possible zircon
    3 = highly probable zircon
    4 = zircon

En DBSCAN se asigna a cada cluster un valor representativo = P90[log10(Zr)]
y se clasifican esos valores por percentiles entre clusters (P75, P90, P95, P99).
En HDBSCAN se usa un score = P90[log10(Zr)] * prob_mean(cluster).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Intentamos importar hdbscan de forma opcional
try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


# ============================================================
# 1. Parámetros globales y paleta 0–4 (tipo "zircon")
# ============================================================

ZIRCON_COLORS = [
    "#4b4b4b",  # 0 no zircon
    "#4aa8ff",  # 1 dubious
    "#3cb371",  # 2 possible
    "#ffa500",  # 3 highly probable
    "#ff0000",  # 4 zircon
]

ZIRCON_LABELS = [
    "no zircon",
    "dubious zircon",
    "possible zircon",
    "highly probable zircon",
    "zircon",
]


# ============================================================
# 2. Utilidades: cargar mapa Zr, construir logZr, features, etc.
# ============================================================

def find_zr_file(mapas_dir):
    """
    Busca un archivo de Zr en la carpeta de mapas:
        - Zr.txt / Zr.tsv
        - *_Zr.txt / *_Zr.tsv
    """
    if not os.path.isdir(mapas_dir):
        raise FileNotFoundError(f"No existe la carpeta de mapas: {mapas_dir}")

    for fname in os.listdir(mapas_dir):
        path = os.path.join(mapas_dir, fname)
        if not os.path.isfile(path):
            continue
        name, ext = os.path.splitext(fname)
        if ext.lower() not in (".txt", ".tsv"):
            continue
        if name == "Zr" or name.endswith("_Zr"):
            return path

    return None


def load_zr_map(mapas_dir):
    """Carga el mapa de Zr como matriz float."""
    zr_path = find_zr_file(mapas_dir)
    if zr_path is None:
        raise FileNotFoundError(
            f"No se encontró Zr.txt ni *_Zr.txt en {mapas_dir}"
        )
    arr = np.loadtxt(zr_path)
    return arr.astype(float), zr_path


def build_logzr(zr):
    """
    Construye log10(Zr) tratando ceros y valores no positivos:
      - Zr > 0  → log10(Zr)
      - Zr <= 0 → se asigna el mínimo log10(Zr) válido de la imagen.
    """
    zr_pos = np.where(zr > 0, zr, np.nan)
    logzr = np.log10(zr_pos)
    if np.isfinite(logzr).any():
        mn = np.nanmin(logzr)
        logzr = np.where(np.isfinite(logzr), logzr, mn)
    else:
        # Caso extremo: no hay valores positivos, devolvemos ceros
        logzr = np.zeros_like(zr)
    return logzr


def build_feature_matrix(logzr):
    """
    Construye el espacio de características X_scaled para DBSCAN/HDBSCAN:

        X = [logZr_norm, x_norm, y_norm]

    donde logZr_norm, x_norm, y_norm son cada uno estándar (media 0, var 1).
    """
    ny, nx = logzr.shape
    logzr_flat = logzr.flatten()

    # Coordenadas x, y
    ys, xs = np.indices((ny, nx))
    xs_flat = xs.flatten().astype(float)
    ys_flat = ys.flatten().astype(float)

    # Matriz de features sin escalar
    X = np.vstack([logzr_flat, xs_flat, ys_flat]).T  # (Npix, 3)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, logzr_flat, (ny, nx)


# ============================================================
# 3. Guardar mapas 0–4 con leyenda tipo zircon
# ============================================================

def save_zircon_class_map(classes_img, out_png, title=None):
    """
    Guarda un mapa de clases 0–4 usando la paleta ZIRCON_COLORS
    y una leyenda compacta.
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.patches as patches

    cmap = ListedColormap(ZIRCON_COLORS)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(classes_img, cmap=cmap, norm=norm)
    ax.axis("off")
    if title:
        ax.set_title(title)

    # Leyenda compacta
    patches_ = [
        patches.Patch(color=c, label=lab)
        for c, lab in zip(ZIRCON_COLORS, ZIRCON_LABELS)
    ]
    fig.legend(handles=patches_, loc="center right",
               frameon=False, borderaxespad=0.3)

    plt.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()


# ============================================================
# 4. De "valor representativo por cluster" a clases 0–4
# ============================================================

def cluster_values_to_classes(cluster_repr):
    """
    Recibe un diccionario:
        cluster_repr = {cluster_id: valor_representativo}
    Por ejemplo:
        - DBSCAN: valor_representativo = P90[log10(Zr)] del cluster
        - HDBSCAN: valor_representativo = P90[log10(Zr)] * prob_mean(cluster)

    Devuelve:
        cluster_class: dict {cluster_id: clase_0_a_4}
        thresholds:    dict con P75, P90, P95, P99
    """
    if not cluster_repr:
        return {}, {}

    vals = np.array(list(cluster_repr.values()), dtype=float)
    P75, P90, P95, P99 = np.percentile(vals, [75, 90, 95, 99])

    thresholds = {"P75": P75, "P90": P90, "P95": P95, "P99": P99}
    cluster_class = {}

    for cid, v in cluster_repr.items():
        if v <= P75:
            c = 0
        elif v <= P90:
            c = 1
        elif v <= P95:
            c = 2
        elif v <= P99:
            c = 3
        else:
            c = 4
        cluster_class[cid] = c

    return cluster_class, thresholds


# ============================================================
# 5. DBSCAN
# ============================================================

def run_dbscan(X_scaled, img_shape, logzr_flat,
               sample_dir, sample,
               eps=0.15, min_samples=30):
    """
    Ejecuta DBSCAN sobre X_scaled y genera:

      - labels imagen (ny x nx)
      - PNG de clusters por ID
      - TSV de labels
      - CSV resumen por cluster
      - Mapa adicional de clases 0–4 (tipo "zircon") + TSV + TXT
    """
    ny, nx = img_shape
    print(">> [DBSCAN] 0%  - iniciando clustering...")
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(X_scaled)

    labels = db.labels_  # -1 = ruido
    labels_img = labels.reshape((ny, nx))

    outdir = os.path.join(sample_dir, f"{sample}_Zr_DBSCAN")
    os.makedirs(outdir, exist_ok=True)

    # 5.1 Guardar TSV de labels
    tsv_path = os.path.join(outdir, f"{sample}_Zr_DBSCAN_labels.tsv")
    np.savetxt(tsv_path, labels_img, fmt="%d", delimiter="\t")

    # 5.2 Mapa de clusters por ID
    plt.figure(figsize=(6, 5))
    vmax = labels[labels >= 0].max() if np.any(labels >= 0) else 0
    im = plt.imshow(labels_img, cmap="tab20", vmin=-1, vmax=max(vmax, 0))
    plt.title(f"DBSCAN clusters (sample={sample})")
    plt.axis("off")
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label("Cluster ID (-1 = ruido)")
    plt.tight_layout()
    png_clusters = os.path.join(outdir, f"{sample}_Zr_DBSCAN_clusters.png")
    plt.savefig(png_clusters, dpi=300)
    plt.close()

    # 5.3 Resumen por cluster (estadística sobre log10(Zr))
    unique_labels = np.unique(labels)
    lines = []
    lines.append("cluster_id,n_pixels,frac_pixels,logZr_mean,logZr_max")
    tot = labels.size

    for cid in unique_labels:
        mask = (labels == cid)
        n = int(mask.sum())
        frac = n / tot if tot > 0 else 0.0
        if n > 0:
            vals = logzr_flat[mask]
            mean_v = float(np.mean(vals))
            max_v = float(np.max(vals))
        else:
            mean_v = max_v = 0.0
        lines.append(f"{cid},{n},{frac:.6f},{mean_v:.6g},{max_v:.6g}")

    csv_path = os.path.join(outdir, f"{sample}_Zr_DBSCAN_summary.csv")
    with open(csv_path, "w") as f:
        f.write("\n".join(lines))

    print(">> [DBSCAN] 60% - resumen de clusters generado.")

    # 5.4 Clasificación 0–4 tipo "zircon" a partir de P90[log10(Zr)] por cluster
    cluster_repr = {}
    for cid in unique_labels:
        if cid < 0:
            continue  # ruido, se gestionará como clase 0
        mask = (labels == cid)
        vals = logzr_flat[mask]
        vals_valid = vals[np.isfinite(vals)]
        if vals_valid.size == 0:
            continue
        rep_val = np.percentile(vals_valid, 90)  # P90 log10(Zr)
        cluster_repr[cid] = rep_val

    cluster_class, thresholds = cluster_values_to_classes(cluster_repr)

    # Mapa de clases 0–4 (ruido = 0)
    classes_img = np.zeros_like(labels_img, dtype=int)
    for cid, c in cluster_class.items():
        classes_img[labels_img == cid] = c
    # Ruido (label = -1) permanece como 0

    # Guardar TSV y PNG
    tsv_classes = os.path.join(outdir, f"{sample}_Zr_DBSCAN_classes.tsv")
    np.savetxt(tsv_classes, classes_img.astype(int), fmt="%d", delimiter="\t")

    png_classes = os.path.join(outdir, f"{sample}_Zr_DBSCAN_classes.png")
    save_zircon_class_map(
        classes_img,
        png_classes,
        title=f"DBSCAN zircon-classes (sample={sample})"
    )

    # TXT de explicación
    txt_path = os.path.join(outdir, f"{sample}_Zr_DBSCAN_stats.txt")
    with open(txt_path, "w") as f:
        f.write("CLASIFICACIÓN 0–4 TIPO 'ZIRCON' (DBSCAN)\n")
        f.write("=========================================\n\n")
        f.write("Definición de clases a partir de P90[log10(Zr)] por cluster:\n")
        f.write("  0 = no zircon            : rep ≤ P75\n")
        f.write("  1 = dubious zircon       : P75 < rep ≤ P90\n")
        f.write("  2 = possible zircon      : P90 < rep ≤ P95\n")
        f.write("  3 = highly probable      : P95 < rep ≤ P99\n")
        f.write("  4 = zircon               : rep > P99\n\n")

        f.write("Umbrales entre clusters (sobre la distribución de valores representativos):\n")
        for k, v in thresholds.items():
            f.write(f"  {k} = {v:.6f}\n")
        f.write("\n")
        f.write("Notas de interpretación:\n")
        f.write("  - Cada cluster espacial se resume por P90[log10(Zr)].\n")
        f.write("  - Los clusters con valores más altos definen los dominios más\n")
        f.write("    enriquecidos en Zr, candidatos preferentes a circón.\n")
        f.write("  - Los píxeles de ruido (cluster = -1) se asignan automáticamente\n")
        f.write("    a la clase 0 (no zircon).\n")

    print(">> [DBSCAN] 100% - mapas y clasificación 0–4 generados.")
    print(f"   [DBSCAN] Salidas principales en:\n"
          f"      {tsv_path}\n"
          f"      {png_clusters}\n"
          f"      {csv_path}\n"
          f"      {tsv_classes}\n"
          f"      {png_classes}\n"
          f"      {txt_path}")


# ============================================================
# 6. HDBSCAN (opcional)
# ============================================================

def run_hdbscan(X_scaled, img_shape, logzr_flat,
                sample_dir, sample,
                min_cluster_size=50, min_samples=20):
    """
    Ejecuta HDBSCAN (si está disponible) y genera:

      - labels imagen (ny x nx)
      - PNG de clusters por ID
      - PNG de probabilidades
      - TSV de labels
      - TSV de probabilidades
      - CSV resumen por cluster (incluyendo prob_mean)
      - Mapa adicional de clases 0–4 (tipo "zircon") + TSV + TXT
    """
    if not HAS_HDBSCAN:
        print(">> [HDBSCAN] hdbscan no está instalado; se omite este análisis.")
        return

    ny, nx = img_shape
    print(">> [HDBSCAN] 0%  - iniciando clustering...")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean'
    )
    clusterer.fit(X_scaled)

    labels = clusterer.labels_            # -1 = ruido
    probs = clusterer.probabilities_      # probabilidad de pertenencia

    labels_img = labels.reshape((ny, nx))
    probs_img = probs.reshape((ny, nx))

    outdir = os.path.join(sample_dir, f"{sample}_Zr_HDBSCAN")
    os.makedirs(outdir, exist_ok=True)

    # 6.1 Guardar TSV de labels y probabilidades
    tsv_labels = os.path.join(outdir, f"{sample}_Zr_HDBSCAN_labels.tsv")
    np.savetxt(tsv_labels, labels_img, fmt="%d", delimiter="\t")

    tsv_probs = os.path.join(outdir, f"{sample}_Zr_HDBSCAN_probabilities.tsv")
    np.savetxt(tsv_probs, probs_img, fmt="%.6f", delimiter="\t")

    # 6.2 Mapas de clusters y de probabilidades
    # Clusters por ID
    plt.figure(figsize=(6, 5))
    vmax = labels[labels >= 0].max() if np.any(labels >= 0) else 0
    im1 = plt.imshow(labels_img, cmap="tab20", vmin=-1, vmax=max(vmax, 0))
    plt.title(f"HDBSCAN clusters (sample={sample})")
    plt.axis("off")
    cbar1 = plt.colorbar(im1, shrink=0.8)
    cbar1.set_label("Cluster ID (-1 = ruido)")
    plt.tight_layout()
    png_clusters = os.path.join(outdir, f"{sample}_Zr_HDBSCAN_clusters.png")
    plt.savefig(png_clusters, dpi=300)
    plt.close()

    # Probabilidades
    plt.figure(figsize=(6, 5))
    im2 = plt.imshow(probs_img, cmap="viridis", vmin=0, vmax=1)
    plt.title(f"HDBSCAN probabilities (sample={sample})")
    plt.axis("off")
    cbar2 = plt.colorbar(im2, shrink=0.8)
    cbar2.set_label("Probabilidad de pertenencia al cluster")
    plt.tight_layout()
    png_probs = os.path.join(outdir, f"{sample}_Zr_HDBSCAN_probabilities.png")
    plt.savefig(png_probs, dpi=300)
    plt.close()

    print(">> [HDBSCAN] 50% - mapas de clusters y probabilidades generados.")

    # 6.3 Resumen por cluster (estadística sobre log10(Zr) + prob_mean)
    unique_labels = np.unique(labels)
    lines = []
    lines.append(
        "cluster_id,n_pixels,frac_pixels,logZr_mean,logZr_max,prob_mean"
    )
    tot = labels.size

    for cid in unique_labels:
        mask = (labels == cid)
        n = int(mask.sum())
        frac = n / tot if tot > 0 else 0.0
        if n > 0:
            vals = logzr_flat[mask]
            mean_v = float(np.mean(vals))
            max_v = float(np.max(vals))
            prob_mean = float(np.mean(probs[mask]))
        else:
            mean_v = max_v = prob_mean = 0.0
        lines.append(
            f"{cid},{n},{frac:.6f},{mean_v:.6g},{max_v:.6g},{prob_mean:.6g}"
        )

    csv_path = os.path.join(outdir, f"{sample}_Zr_HDBSCAN_summary.csv")
    with open(csv_path, "w") as f:
        f.write("\n".join(lines))

    print(">> [HDBSCAN] 75% - resumen de clusters generado.")

    # 6.4 Clasificación 0–4 tipo "zircon" a partir de:
    #       score_cluster = P90[log10(Zr)] * prob_mean(cluster)
    cluster_repr = {}
    for cid in unique_labels:
        if cid < 0:
            continue  # ruido
        mask = (labels == cid)
        vals = logzr_flat[mask]
        vals_valid = vals[np.isfinite(vals)]
        if vals_valid.size == 0:
            continue
        logZr_p90 = np.percentile(vals_valid, 90)
        prob_mean = float(np.mean(probs[mask]))
        rep_val = logZr_p90 * prob_mean
        cluster_repr[cid] = rep_val

    cluster_class, thresholds = cluster_values_to_classes(cluster_repr)

    classes_img = np.zeros_like(labels_img, dtype=int)
    for cid, c in cluster_class.items():
        classes_img[labels_img == cid] = c
    # Ruido (label = -1) permanece como clase 0

    tsv_classes = os.path.join(outdir, f"{sample}_Zr_HDBSCAN_classes.tsv")
    np.savetxt(tsv_classes, classes_img.astype(int), fmt="%d", delimiter="\t")

    png_classes = os.path.join(outdir, f"{sample}_Zr_HDBSCAN_classes.png")
    save_zircon_class_map(
        classes_img,
        png_classes,
        title=f"HDBSCAN zircon-classes (sample={sample})"
    )

    txt_path = os.path.join(outdir, f"{sample}_Zr_HDBSCAN_stats.txt")
    with open(txt_path, "w") as f:
        f.write("CLASIFICACIÓN 0–4 TIPO 'ZIRCON' (HDBSCAN)\n")
        f.write("==========================================\n\n")
        f.write("Score de cada cluster = P90[log10(Zr)] × prob_mean(cluster).\n")
        f.write("Definición de clases a partir de los percentiles P75–P90–P95–P99 del score:\n")
        f.write("  0 = no zircon            : score ≤ P75\n")
        f.write("  1 = dubious zircon       : P75 < score ≤ P90\n")
        f.write("  2 = possible zircon      : P90 < score ≤ P95\n")
        f.write("  3 = highly probable      : P95 < score ≤ P99\n")
        f.write("  4 = zircon               : score > P99\n\n")

        f.write("Umbrales entre clusters (sobre la distribución de scores):\n")
        for k, v in thresholds.items():
            f.write(f"  {k} = {v:.6f}\n")
        f.write("\n")
        f.write("Notas de interpretación:\n")
        f.write("  - HDBSCAN permite detectar clusters de diferente densidad y\n")
        f.write("    aporta una probabilidad de pertenencia para cada píxel.\n")
        f.write("  - Al combinar intensidad (P90 log10(Zr)) y prob_mean, se\n")
        f.write("    favorecen los clusters compactos y coherentes, típicos de\n")
        f.write("    granos discretos de circón.\n")
        f.write("  - Los píxeles de ruido (label = -1) se asignan a la clase 0.\n")

    print(">> [HDBSCAN] 100% - mapas y clasificación 0–4 generados.")
    print(f"   [HDBSCAN] Salidas principales en:\n"
          f"      {tsv_labels}\n"
          f"      {tsv_probs}\n"
          f"      {png_clusters}\n"
          f"      {png_probs}\n"
          f"      {csv_path}\n"
          f"      {tsv_classes}\n"
          f"      {png_classes}\n"
          f"      {txt_path}")


# ============================================================
# 7. MAIN
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("USO: python3 zircon_zr_DBSCAN_HDBSCAN.py <muestra>")
        sys.exit(1)

    sample = sys.argv[1]
    root = os.path.abspath(os.getcwd())
    sample_dir = os.path.join(root, sample)
    mapas_dir = os.path.join(sample_dir, f"{sample}_mapas_tsv")

    if not os.path.isdir(sample_dir):
        print(f"[ERROR] No existe la carpeta de la muestra: {sample_dir}")
        sys.exit(1)

    if not os.path.isdir(mapas_dir):
        print(f"[ERROR] No existe la carpeta de mapas: {mapas_dir}")
        sys.exit(1)

    print(f">> Muestra: {sample}")
    print(f">> Carpeta de mapas: {mapas_dir}")

    # 1) Cargar Zr y construir log10(Zr)
    zr, zr_path = load_zr_map(mapas_dir)
    print(f">> Mapa de Zr cargado desde: {zr_path}")
    logzr = build_logzr(zr)

    # 2) Construir matriz de características
    X_scaled, logzr_flat, img_shape = build_feature_matrix(logzr)
    print(">> Espacio de características (logZr, x, y) construido.")

    # 3) Ejecutar DBSCAN
    run_dbscan(
        X_scaled,
        img_shape,
        logzr_flat,
        sample_dir,
        sample,
        eps=0.15,
        min_samples=30
    )

    # 4) Ejecutar HDBSCAN (si está disponible)
    run_hdbscan(
        X_scaled,
        img_shape,
        logzr_flat,
        sample_dir,
        sample,
        min_cluster_size=50,
        min_samples=20
    )

    print(">> TODO LISTO (DBSCAN/HDBSCAN).")


if __name__ == "__main__":
    main()
