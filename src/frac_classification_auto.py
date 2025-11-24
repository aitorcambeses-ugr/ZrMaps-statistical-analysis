#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
frac_classification_auto.py

Clasificación de fases a partir de mapas de intensidad (TSV/TXT)
mediante fracciones normalizadas (FRAC_X) y KMeans (k=15).

- Lee mapas de 10 elementos objetivo: Si, Ti, Al, Fe, Mg, Ca, Na, K, P, Zr
  desde la carpeta: <SAMPLE>/<SAMPLE>_mapas_tsv
- Si falta alguno de estos elementos, se usa un mapa de ceros para ese elemento
  (FRAC correspondiente será ≈0), de modo que el script SIEMPRE funciona
  aunque no haya mapas de todos los elementos.

- Calcula, para cada píxel:
      FRAC_X = I_X / (I_Si + I_Ti + I_Al + I_Fe + I_Mg +
                      I_Ca + I_Na + I_K + I_P + I_Zr)

- Aplica KMeans (k=15) sobre los vectores [FRAC_Si ... FRAC_Zr]
- Genera:

  1) Carpeta de salida:
         <SAMPLE>/<SAMPLE>_FRAC_clusters/

  2) Ficheros dentro de esa carpeta:
     - <SAMPLE>_phase_mask.tsv      → matriz 2D con etiquetas de fase (1–15)
     - <SAMPLE>_phase_mask.png      → imagen de fases (SIN LEYENDA)
     - label_to_phase_map.csv       → estadísticas por fase (CSV)
     - label_to_phase_map.xlsx      → estadísticas por fase (Excel)
     - <SAMPLE>_summary.txt         → informe de texto con FRAC y explicación

Autor: Aitor Cambeses (UGR) + ChatGPT
Versión: 0.3 (FRAC para 10 elementos, summary detallado, PNG sin leyenda)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from openpyxl import Workbook


# ============================================================
# 1. UTILIDADES PARA CARGAR MAPAS
# ============================================================

TARGET_ELEMENTS = ["Si", "Ti", "Al", "Fe", "Mg", "Ca", "Na", "K", "P", "Zr"]


def load_element_map(mapas_dir, elem):
    """
    Intenta cargar el mapa de un elemento desde mapas_dir.

    Busca ficheros:
      - "<elem>.txt"
      - "<elem>.tsv"

    Si NO encuentra nada, devuelve None.
    """
    candidates = [
        f"{elem}.txt",
        f"{elem}.tsv",
    ]
    for fname in candidates:
        path = os.path.join(mapas_dir, fname)
        if os.path.isfile(path):
            arr = np.loadtxt(path)
            return arr.astype(float)
    return None


def load_all_maps(mapas_dir):
    """
    Carga todos los mapas de los elementos TARGET_ELEMENTS.

    - Necesita al menos un mapa real para definir dimensiones (ny, nx).
    - Para elementos sin fichero, crea una matriz de ceros con (ny, nx).
    - Devuelve:
        element_list  : lista de nombres de elementos (10)
        maps_stack    : array de forma (n_elems, ny, nx)
        present_flags : dict elem -> True/False (si existía fichero real)
    """
    # Primero localizamos al menos un elemento real para fijar la forma
    shape = None
    present_flags = {}

    temp_maps = {}

    for elem in TARGET_ELEMENTS:
        arr = load_element_map(mapas_dir, elem)
        if arr is not None:
            temp_maps[elem] = arr
            present_flags[elem] = True
            if shape is None:
                shape = arr.shape

    if shape is None:
        raise FileNotFoundError(
            f"No se encontró ningún mapa de {TARGET_ELEMENTS} en {mapas_dir}"
        )

    ny, nx = shape

    # Ahora garantizamos un mapa para cada elemento; si falta, usamos ceros
    maps = []
    for elem in TARGET_ELEMENTS:
        if elem in temp_maps:
            arr = temp_maps[elem]
            if arr.shape != shape:
                raise ValueError(
                    f"El mapa de {elem} tiene forma {arr.shape}, "
                    f"pero se esperaba {shape}."
                )
        else:
            # Mapa faltante → matriz de ceros
            arr = np.zeros((ny, nx), dtype=float)
            present_flags[elem] = False
        maps.append(arr)

    maps_stack = np.stack(maps, axis=0)  # (n_elems, ny, nx)
    return TARGET_ELEMENTS, maps_stack, present_flags


# ============================================================
# 2. CÁLCULO DE FRAC_X
# ============================================================

def compute_FRAC_maps(maps_stack):
    """
    A partir de maps_stack de forma (n_elems, ny, nx) calcula FRAC_X:

        FRAC_X = I_X / sum(I_all)

    Manejo de ceros:
      - Si la suma de intensidades de un píxel es 0, se pone FRAC_X = 0
        para todos los elementos en ese píxel.
    """
    # maps_stack: (n_elems, ny, nx)
    sum_all = np.sum(maps_stack, axis=0)  # (ny, nx)
    # Evitar división por cero
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_stack = maps_stack / np.where(sum_all == 0, 1.0, sum_all)
    # Donde sum_all == 0 → FRAC=0
    frac_stack[:, sum_all == 0] = 0.0
    return frac_stack  # (n_elems, ny, nx)


# ============================================================
# 3. KMEANS Y GENERACIÓN DE MÁSCARA
# ============================================================

def run_kmeans(frac_stack, n_clusters=15, random_state=0):
    """
    Ejecuta KMeans sobre los vectores de FRAC por píxel.

    Entrada:
      frac_stack: (n_elems, ny, nx)

    Devuelve:
      labels_2d: matriz (ny, nx) con etiquetas 1..n_clusters
      km       : objeto KMeans (por si se quiere inspeccionar)
    """
    n_elems, ny, nx = frac_stack.shape
    # Reshape a (N_pix, n_elems)
    frac_flat = frac_stack.reshape(n_elems, -1).T  # (N_pix, n_elems)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(frac_flat)
    labels = km.labels_  # 0..n_clusters-1

    labels_2d = labels.reshape(ny, nx) + 1  # 1..n_clusters
    return labels_2d, km


# ============================================================
# 4. GUARDAR IMAGEN PNG (SIN LEYENDA)
# ============================================================

def save_phase_png(labels_2d, out_png):
    """
    Guarda la máscara de fases como PNG sin leyenda.
    Cada fase (1..k) se colorea con una paleta categórica sencilla.
    """
    k = int(labels_2d.max())
    ny, nx = labels_2d.shape

    # Paleta simple: repetimos colores si k > len(colors)
    base_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    ]
    colors = base_colors * ((k // len(base_colors)) + 1)
    colors = colors[:k]

    # Creamos un mapa de colores discreto
    from matplotlib.colors import ListedColormap, BoundaryNorm

    cmap = ListedColormap(colors)
    bounds = np.arange(0.5, k + 1.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(labels_2d, cmap=cmap, norm=norm)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


# ============================================================
# 5. GUARDAR TSV, CSV y XLSX
# ============================================================

def save_phase_mask_tsv(labels_2d, out_tsv):
    """Guarda la máscara de fases como TSV (enteros)."""
    np.savetxt(out_tsv, labels_2d.astype(int), fmt="%d", delimiter="\t")


def compute_phase_stats(labels_2d, frac_stack, elements):
    """
    Calcula estadísticas por fase:

      - n_pixels
      - frac_pixels
      - mean FRAC_X (para cada elemento)

    Devuelve:
      stats_list: lista de dicts, uno por fase (1..k)
    """
    ny, nx = labels_2d.shape
    n_elems = len(elements)
    total_pixels = ny * nx

    frac_flat = frac_stack.reshape(n_elems, -1).T  # (N_pix, n_elems)

    k = int(labels_2d.max())
    labels_flat = labels_2d.flatten()

    stats_list = []
    for phase_id in range(1, k + 1):
        mask = (labels_flat == phase_id)
        n_pix = int(mask.sum())
        if n_pix > 0:
            mean_fracs = np.mean(frac_flat[mask, :], axis=0)
        else:
            mean_fracs = np.zeros(n_elems, dtype=float)

        stats = {
            "phase_id": phase_id,
            "n_pixels": n_pix,
            "frac_pixels": n_pix / total_pixels if total_pixels > 0 else 0.0,
            "mean_fracs": {elements[i]: float(mean_fracs[i]) for i in range(n_elems)},
        }
        stats_list.append(stats)

    return stats_list


def guess_phase_interpretation(mean_fracs):
    """
    Heurística MUY simple para sugerir una interpretación mineral aproximada.
    No sustituye la interpretación petrográfica.

    mean_fracs: dict {elem: FRAC_elem}
    """
    Si = mean_fracs.get("Si", 0.0)
    Al = mean_fracs.get("Al", 0.0)
    Fe = mean_fracs.get("Fe", 0.0)
    Mg = mean_fracs.get("Mg", 0.0)
    Ca = mean_fracs.get("Ca", 0.0)
    Na = mean_fracs.get("Na", 0.0)
    K  = mean_fracs.get("K",  0.0)
    P  = mean_fracs.get("P",  0.0)
    Ti = mean_fracs.get("Ti", 0.0)
    Zr = mean_fracs.get("Zr", 0.0)

    # Algunos umbrales relativos muy sencillos
    if Zr > 0.1 and Si > 0.1:
        return "Dominantly Zr–Si rich: probable zircon / Zr-rich accessory"

    if P > 0.1 and Ca > 0.1:
        return "Ca–P rich: probable apatite / phosphate"

    if (Fe + Mg) > 0.3 and Si > 0.1:
        return "Fe–Mg–Si rich: mafic silicate (amphibole/pyroxene/biotite)"

    if (Si + Al) > 0.6 and (Na + Ca + K) > 0.2:
        return "Si–Al + alkalis: feldspar (plagioclase / K-feldspar domain)"

    if Ti > 0.1:
        return "Ti-rich domain: Fe–Ti oxides / titanite"

    return "Mixed composition: requires petrographic/contextual interpretation"


def save_label_to_phase_csv(stats_list, elements, out_csv):
    """
    Guarda label_to_phase_map.csv con columnas:

      phase_id, n_pixels, frac_pixels, FRAC_Si, FRAC_Ti, ..., FRAC_Zr
    """
    header = ["phase_id", "n_pixels", "frac_pixels"] + [
        f"FRAC_{e}" for e in elements
    ]
    lines = [",".join(header)]

    for st in stats_list:
        row = [str(st["phase_id"]),
               str(st["n_pixels"]),
               f"{st['frac_pixels']:.6f}"]
        for e in elements:
            row.append(f"{st['mean_fracs'][e]:.6f}")
        lines.append(",".join(row))

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_label_to_phase_xlsx(stats_list, elements, out_xlsx):
    """
    Guarda label_to_phase_map.xlsx con las mismas columnas que el CSV.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "label_to_phase"

    header = ["phase_id", "n_pixels", "frac_pixels"] + [
        f"FRAC_{e}" for e in elements
    ]
    ws.append(header)

    for st in stats_list:
        row = [st["phase_id"], st["n_pixels"], st["frac_pixels"]]
        for e in elements:
            row.append(st["mean_fracs"][e])
        ws.append(row)

    wb.save(out_xlsx)


def save_summary_txt(sample, labels_2d, stats_list, elements,
                     present_flags, out_txt):
    """
    Genera un informe de texto con:

      - Sample, dimensiones, número de píxeles
      - Elementos usados
      - Estadística por fase (n_pix, frac_pix, FRAC_X ordenadas)
      - Definición de FRAC
      - Explicación del método y notas de interpretación
    """
    ny, nx = labels_2d.shape
    total_pixels = ny * nx

    lines = []
    lines.append(f"Sample: {sample}")
    lines.append(f"Image shape (ny, nx): {ny}, {nx}")
    lines.append(f"Total pixels: {total_pixels}")
    lines.append("")

    # Elementos usados
    lines.append("Elements used in FRAC computation:")
    for e in elements:
        flag = present_flags.get(e, False)
        if flag:
            lines.append(f"  {e}  (map present)")
        else:
            lines.append(f"  {e}  (no map found → intensities = 0)")
    lines.append("")

    # Estadística por fase
    k = len(stats_list)
    lines.append(f"Phase statistics (KMeans k={k}, labels 1–{k}):")
    lines.append("")

    for st in stats_list:
        pid = st["phase_id"]
        lines.append(f"Phase {pid:2d}:")
        lines.append(f"  n_pixels   = {st['n_pixels']}")
        lines.append(f"  frac_pixels= {st['frac_pixels']*100:.3f} %")

        # Ordenar FRAC de mayor a menor
        mean_fracs = st["mean_fracs"]
        sorted_fracs = sorted(mean_fracs.items(),
                              key=lambda kv: kv[1],
                              reverse=True)
        lines.append("  Mean FRAC per element (descending):")
        for elem, v in sorted_fracs:
            lines.append(f"    FRAC_{elem:2s} = {v:.6f}")

        # Interpretación heurística
        interp = guess_phase_interpretation(mean_fracs)
        lines.append(f"  Approximate interpretation (heuristic): {interp}")
        lines.append("")

    # Bloque explicativo
    lines.append("DEFINITION OF FRACTIONS (FRAC_X)")
    lines.append("For each pixel, FRAC_X is defined as:")
    lines.append("")
    lines.append("  FRAC_X = I_X / (I_Si + I_Ti + I_Al + I_Fe + I_Mg +")
    lines.append("                  I_Ca + I_Na + I_K + I_P + I_Zr)")
    lines.append("")
    lines.append("where I_X is the intensity of element X in that pixel")
    lines.append("and the denominator is the sum of intensities of all")
    lines.append("10 elements considered in this workflow.")
    lines.append("")
    lines.append("These fractions are dimensionless and represent the")
    lines.append("relative contribution of each element in each pixel,")
    lines.append("allowing composition comparison independently of the")
    lines.append("absolute brightness of the map.")
    lines.append("")
    lines.append("If one of the target elements does not have an input map,")
    lines.append("its intensity is assumed to be zero and its FRAC_X values")
    lines.append("are therefore ~0 in all phases.")
    lines.append("")
    lines.append("METHOD: K-MEANS PHASE CLASSIFICATION")
    lines.append("")
    lines.append("The phases are obtained by applying the KMeans algorithm")
    lines.append("with k=15 to the normalized fraction vectors:")
    lines.append("")
    lines.append("  [FRAC_Si, FRAC_Ti, FRAC_Al, FRAC_Fe, FRAC_Mg,")
    lines.append("   FRAC_Ca, FRAC_Na, FRAC_K, FRAC_P, FRAC_Zr]")
    lines.append("")
    lines.append("Each phase corresponds to one cluster centre in this")
    lines.append("multidimensional space. The phase labels (1–15) are")
    lines.append("purely statistical and do not imply fixed mineral names.")
    lines.append("")
    lines.append("INTERPRETATION OF PHASES (CAUTION)")
    lines.append("")
    lines.append("- The approximate interpretations printed above are")
    lines.append("  heuristic and should be confirmed using petrography,")
    lines.append("  BSE images, and (where available) quantitative")
    lines.append("  microanalysis (EPMA/WDS, SEM–EDS, etc.).")
    lines.append("- The FRAC-based classification is very useful to:")
    lines.append("    * Highlight zircon-rich domains (Zr + Si enrichment)")
    lines.append("    * Distinguish Ca–P rich apatite from silicate matrix")
    lines.append("    * Separate mafic vs felsic silicates")
    lines.append("    * Locate Fe–Ti oxide/titanite clusters")
    lines.append("")
    lines.append("This summary file is intended as a quick, human-readable")
    lines.append("overview to guide interpretation and selection of phases")
    lines.append("for further detailed investigation (e.g., zircon picking")
    lines.append("prior to SHRIMP/LA-ICP-MS, or selection of host minerals")
    lines.append("for barometry, thermometry, and diffusion studies).")
    lines.append("")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# 6. MAIN
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("USO: python frac_classification_auto.py <SAMPLE>")
        sys.exit(1)

    sample = sys.argv[1]
    root = os.path.abspath(os.getcwd())
    sample_dir = os.path.join(root, sample)
    mapas_dir = os.path.join(sample_dir, f"{sample}_mapas_tsv")

    if not os.path.isdir(mapas_dir):
        print(f"[ERROR] No existe la carpeta de mapas: {mapas_dir}")
        sys.exit(1)

    print(f">> Sample: {sample}")
    print(f">> Map folder: {mapas_dir}")

    # 1) Cargar mapas de los 10 elementos (o ceros si faltan).
    elements, maps_stack, present_flags = load_all_maps(mapas_dir)
    n_elems, ny, nx = maps_stack.shape
    print(f"   Loaded maps for elements: {elements}")
    print(f"   Image shape: {ny} x {nx}")

    # 2) Calcular FRAC_X
    frac_stack = compute_FRAC_maps(maps_stack)
    print("   FRAC maps computed.")

    # 3) KMeans
    labels_2d, km = run_kmeans(frac_stack, n_clusters=15, random_state=0)
    print("   KMeans clustering completed (k=15).")

    # 4) Crear carpeta de salida
    outdir = os.path.join(sample_dir, f"{sample}_FRAC_clusters")
    os.makedirs(outdir, exist_ok=True)

    # 5) Guardar máscara TSV
    mask_tsv = os.path.join(outdir, f"{sample}_phase_mask.tsv")
    save_phase_mask_tsv(labels_2d, mask_tsv)
    print(f"   Saved phase mask TSV: {mask_tsv}")

    # 6) Guardar PNG SIN leyenda
    mask_png = os.path.join(outdir, f"{sample}_phase_mask.png")
    save_phase_png(labels_2d, mask_png)
    print(f"   Saved phase mask PNG (no legend): {mask_png}")

    # 7) Estadísticas por fase
    stats_list = compute_phase_stats(labels_2d, frac_stack, elements)

    # 8) label_to_phase_map.csv
    csv_path = os.path.join(outdir, "label_to_phase_map.csv")
    save_label_to_phase_csv(stats_list, elements, csv_path)
    print(f"   Saved phase map CSV: {csv_path}")

    # 9) label_to_phase_map.xlsx
    xlsx_path = os.path.join(outdir, "label_to_phase_map.xlsx")
    save_label_to_phase_xlsx(stats_list, elements, xlsx_path)
    print(f"   Saved phase map XLSX: {xlsx_path}")

    # 10) summary.txt
    summary_path = os.path.join(outdir, f"{sample}_summary.txt")
    save_summary_txt(sample, labels_2d, stats_list, elements,
                     present_flags, summary_path)
    print(f"   Saved text summary: {summary_path}")

    print(">> FRAC + KMeans phase classification COMPLETED.")


if __name__ == "__main__":
    main()
