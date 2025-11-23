#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
frac_classification_auto.py — fases numeradas 1–15

Clasificación automática de fases (FRAC + KMeans) a partir de 10 mapas
elementales de micro-XRF:

    Si, Ti, Al, Fe, Mg, Ca, Na, K, P, Zr

Requisitos:
    - numpy
    - matplotlib
    - scikit-learn
    - openpyxl  (opcional, sólo para el XLSX)

Estructura de carpetas (ejemplo SAMPLE = "test"):

    micro_XRF/
        frac_classification_auto.py
        SAMPLE/
            SAMPLE_mapas_tsv/
                Si.txt, Ti.txt, Al.txt, Fe.txt, Mg.txt,
                Ca.txt, Na.txt, K.txt, P.txt, Zr.txt

Salida:

    SAMPLE/SAMPLE_FRAC_clusters/
        SAMPLE_phase_mask.tsv
        SAMPLE_phase_mask.png         (fases 1–15, colormap 'rainbow')
        label_to_phase_map.csv        (FRAC + ratios por fase)
        label_to_phase_map.xlsx       (si hay openpyxl)
        SAMPLE_summary.txt            (definición FRAC + fases 1–15
                                       con interpretación aproximada)
"""

import os
import sys
import numpy as np

ELEMENTS = ["Si", "Ti", "Al", "Fe", "Mg", "Ca", "Na", "K", "P", "Zr"]


# ----------------------------------------------------------------------
# Utilidades de lectura
# ----------------------------------------------------------------------
def find_element_file(folder: str, element: str):
    """
    Busca un archivo para un elemento dentro de 'folder'.
    Acepta:
        element.txt / element.tsv
        *_element.txt / *_element.tsv
    Devuelve la ruta completa o None si no encuentra nada.
    """
    candidates = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            continue
        name, ext = os.path.splitext(fname)
        ext = ext.lower()
        if ext not in (".tsv", ".txt"):
            continue
        if name == element or name.endswith("_" + element):
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort()
    return candidates[0]


def load_maps(folder: str):
    """
    Carga todos los mapas definidos en ELEMENTS desde 'folder'.
    Asume que los ficheros NO tienen cabecera, solo matriz numérica.
    """
    maps = {}
    ref_shape = None
    for el in ELEMENTS:
        fpath = find_element_file(folder, el)
        if fpath is None:
            raise FileNotFoundError(
                f"No se encontró archivo para '{el}' en {folder}"
            )
        arr = np.loadtxt(fpath)
        if ref_shape is None:
            ref_shape = arr.shape
        else:
            if arr.shape != ref_shape:
                raise ValueError(
                    f"Dimensiones incompatibles en {fpath}: {arr.shape} vs {ref_shape}"
                )
        maps[el] = arr.astype(float)
    return maps, ref_shape


def build_frac_stack(maps, shape):
    """
    Construye el cubo FRAC HxWxN a partir de los mapas.
    Cada píxel se normaliza a suma 1 (solo relaciones relativas).

    FRAC_X = I_X / sum(I_elementos)
    """
    h, w = shape
    n = len(ELEMENTS)
    stack = np.zeros((h, w, n), dtype=float)
    for i, el in enumerate(ELEMENTS):
        # Evitar negativos raros
        m = np.maximum(maps[el], 0.0)
        stack[:, :, i] = m

    total = np.sum(stack, axis=-1, keepdims=True)
    total[total == 0] = 1.0      # evita división por cero
    frac = stack / total
    return frac


# ----------------------------------------------------------------------
# KMeans
# ----------------------------------------------------------------------
def run_kmeans(frac, k: int = 15, random_state: int = 0):
    """
    Ejecuta KMeans sobre el cubo FRAC.
    Devuelve:
        labels_img: matriz HxW con ID de cluster (0..k-1)
        km: objeto KMeans ajustado
    """
    from sklearn.cluster import KMeans

    h, w, n = frac.shape
    X = frac.reshape(-1, n)

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    labels_img = labels.reshape(h, w)
    return labels_img, km


# ----------------------------------------------------------------------
# Guardado de mapas de fases
# ----------------------------------------------------------------------
def save_phase_map_tsv(labels_img, path: str):
    """
    Guarda la matriz de fases en TSV.
    """
    np.savetxt(path, labels_img.astype(int), fmt="%d", delimiter="\t")


def save_phase_map_png(labels_img, path: str):
    """
    Guarda un PNG de la máscara de fases usando colormap 'rainbow'.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=labels_img.min(), vmax=labels_img.max())
    rgba = cm.rainbow(norm(labels_img))

    plt.figure(figsize=(6, 6))
    plt.imshow(rgba, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


# ----------------------------------------------------------------------
# Cálculo de centros y ratios por fase
# ----------------------------------------------------------------------
def compute_centers_and_ratios(km):
    """
    A partir de los centros de KMeans (FRAC_), calcula también 4 ratios derivadas.

    Devuelve:
        header -> lista de nombres de columnas
        rows   -> lista de filas [cluster_id, FRAC_..., FR_...]
    """
    centers = km.cluster_centers_
    k, n = centers.shape

    frac_headers = [f"FRAC_{el}" for el in ELEMENTS]

    header = (
        ["cluster_id"]
        + frac_headers
        + ["FR_SiAl", "FR_Fe_FeMg", "FR_Ca_NaK", "FR_Si_SiZr"]
    )

    rows = []
    eps = 1e-12

    i_Si = ELEMENTS.index("Si")
    i_Al = ELEMENTS.index("Al")
    i_Fe = ELEMENTS.index("Fe")
    i_Mg = ELEMENTS.index("Mg")
    i_Ca = ELEMENTS.index("Ca")
    i_Na = ELEMENTS.index("Na")
    i_K = ELEMENTS.index("K")
    i_Zr = ELEMENTS.index("Zr")

    for cid in range(k):
        c = centers[cid, :]  # vector de FRAC_
        frac_values = list(c)

        FR_SiAl = c[i_Si] + c[i_Al]
        FR_Fe_FeMg = c[i_Fe] / (c[i_Fe] + c[i_Mg] + eps)
        FR_Ca_NaK = c[i_Ca] / (c[i_Na] + c[i_K] + eps)
        FR_Si_SiZr = c[i_Si] / (c[i_Si] + c[i_Zr] + eps)

        # cluster_id numerado de 1 a k
        row = [cid + 1] + frac_values + [
            FR_SiAl,
            FR_Fe_FeMg,
            FR_Ca_NaK,
            FR_Si_SiZr,
        ]
        rows.append(row)

    return header, rows


def save_label_stats_and_excel(km, csv_path: str):
    """
    Guarda:
      - CSV con FRAC_elemento medio por cluster + 4 ratios derivadas
      - Excel (.xlsx) con la misma tabla (usando openpyxl si está disponible)
    """
    header, rows = compute_centers_and_ratios(km)

    # CSV
    lines = [",".join(header)]
    for row in rows:
        cid = row[0]
        values = row[1:]
        line = ",".join([str(cid)] + [f"{v:.6f}" for v in values])
        lines.append(line)

    with open(csv_path, "w") as f:
        f.write("\n".join(lines))

    # Excel
    xlsx_path = os.path.splitext(csv_path)[0] + ".xlsx"
    try:
        from openpyxl import Workbook  # type: ignore

        wb = Workbook()
        ws = wb.active
        ws.title = "FRAC_phases"
        ws.append(header)
        for row in rows:
            ws.append(row)
        wb.save(xlsx_path)
    except ImportError:
        xlsx_path = None

    return xlsx_path


# ----------------------------------------------------------------------
# Interpretación mineralógica aproximada
# ----------------------------------------------------------------------
def guess_mineral_from_fracs(frac_dict):
    """
    Asignación mineralógica simple (opción B) en función de las FRAC principales.
    Esta interpretación es aproximada y sirve solo como guía rápida.
    """
    si = frac_dict["Si"]
    ti = frac_dict["Ti"]
    al = frac_dict["Al"]
    fe = frac_dict["Fe"]
    mg = frac_dict["Mg"]
    ca = frac_dict["Ca"]
    na = frac_dict["Na"]
    k = frac_dict["K"]
    p = frac_dict["P"]
    zr = frac_dict["Zr"]

    # Zircon / Zr-rich
    if zr > 0.02 and zr == max(frac_dict.values()):
        return "Fase rica en Zr (probable circón u otro accesorio Zr)"

    # Cuarzo casi puro
    if si > 0.90 and al < 0.05 and (fe + mg + ca + na + k + p + ti + zr) < 0.10:
        return "Fase rica en Si (probable cuarzo)"

    # Feldespatos
    if si > 0.45 and al > 0.20 and (na + k + ca) > 0.05:
        if k > na and k > ca:
            return "Fase Si–Al–K (probable K-feldespato o mica potásica)"
        if na >= ca and na > k:
            return "Fase Si–Al–Na (probable plagioclasa sodica/albítica)"
        if ca > na and ca > k:
            return "Fase Si–Al–Ca (probable plagioclasa cálcica)"
        return "Fase Si–Al–(Na,K,Ca) (probable feldespato)"

    # Máficas Fe–Mg
    if (fe + mg) > 0.25 and si > 0.30:
        if fe > mg:
            return "Fase máfica Fe-rica (probable biotita/anfíbol Fe-rico)"
        else:
            return "Fase máfica Mg-rica (probable olivino/px/anfíbol Mg-rico)"

    # Óxidos Fe–Ti
    if (fe + ti) > 0.25 and si < 0.30:
        return "Fase óxidos Fe–Ti / titanita"

    # Fosfatos
    if p > 0.02:
        return "Fase rica en P (probable apatito/fosfato)"

    # Default
    return "Fase mixta / sin asignación clara (requiere inspección adicional)"


# ----------------------------------------------------------------------
# Resumen en texto
# ----------------------------------------------------------------------
def build_summary_text(km):
    """
    Construye el texto del resumen:
    - Definición de FRAC_X
    - Descripción del método
    - Tabla simple por fase con elementos dominantes e interpretación aproximada.
    """
    centers = km.cluster_centers_
    k, n = centers.shape

    lines = []
    lines.append(f"Numero de fases (clusters) identificadas: {k}")
    lines.append("")
    lines.append("Las fases están numeradas de 1 a 15.")
    lines.append("")
    lines.append("DEFINICION DE LAS FRACCIONES (FRAC_X)")
    lines.append(
        "Para cada pixel se define FRAC_X como la intensidad del elemento X "
        "dividida por la suma total de intensidades de los 10 elementos considerados:"
    )
    lines.append(
        "FRAC_X = I_X / (I_Si + I_Ti + I_Al + I_Fe + I_Mg + I_Ca + I_Na + I_K + I_P + I_Zr)"
    )
    lines.append(
        "Estas fracciones son adimensionales y representan la contribucion relativa de "
        "cada elemento en el pixel, permitiendo comparar composiciones sin depender "
        "del brillo absoluto del mapa."
    )
    lines.append("")
    lines.append("METODO DE ASIGNACION DE FASES")
    lines.append(
        "Las fases se obtienen aplicando el algoritmo K-means (k=15) sobre el vector "
        "de fracciones normalizadas [FRAC_Si, FRAC_Ti, FRAC_Al, FRAC_Fe, FRAC_Mg, "
        "FRAC_Ca, FRAC_Na, FRAC_K, FRAC_P, FRAC_Zr] de cada pixel. "
        "Cada fase corresponde a uno de los centros de cluster en este espacio "
        "multidimensional."
    )
    lines.append("")
    lines.append("RESUMEN QUIMICO E INTERPRETACION APROXIMADA POR FASE")
    lines.append(
        "(Los valores mostrados son las fracciones medias FRAC_X por fase; "
        "la interpretacion mineralogica es orientativa.)"
    )
    lines.append("")

    for cid in range(k):
        c = centers[cid, :]
        frac_dict = {el: c[i] for i, el in enumerate(ELEMENTS)}

        sorted_elems = sorted(
            frac_dict.items(), key=lambda kv: kv[1], reverse=True
        )
        top3 = sorted_elems[:3]

        interp = guess_mineral_from_fracs(frac_dict)

        lines.append(f"Fase {cid + 1}:")
        dom_str = ", ".join([f"{el}={val:.3f}" for el, val in top3])
        lines.append(f"  Elementos dominantes (FRAC_X): {dom_str}")
        lines.append(f"  Interpretacion aproximada: {interp}")
        lines.append("")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# main()
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("USO: python3 frac_classification_auto.py <muestra>")
        sys.exit(1)

    sample = sys.argv[1]
    root = os.path.abspath(os.getcwd())
    sample_dir = os.path.join(root, sample)
    if not os.path.isdir(sample_dir):
        print(f"[ERROR] No existe la carpeta de muestra: {sample_dir}")
        sys.exit(1)

    mapas_dir = os.path.join(sample_dir, f"{sample}_mapas_tsv")
    if not os.path.isdir(mapas_dir):
        print(f"[ERROR] Falta la carpeta de mapas: {mapas_dir}")
        sys.exit(1)

    out_dir = os.path.join(sample_dir, f"{sample}_FRAC_clusters")
    os.makedirs(out_dir, exist_ok=True)

    print(f">> Muestra: {sample}")
    print(f">> Carpeta de mapas: {mapas_dir}")
    print(f">> Carpeta de salida: {out_dir}")

    # 1) Cargar mapas
    print(">> Cargando mapas...")
    maps, shape = load_maps(mapas_dir)

    # 2) Construir fracciones
    print(">> Construyendo fracciones (FRAC_)...")
    frac = build_frac_stack(maps, shape)

    # 3) KMeans
    print(">> Ejecutando KMeans (k=15)...")
    labels_img, km = run_kmeans(frac, k=15)

    # IMPORTANTE: numerar fases 1–15 en la máscara de salida
    labels_img = labels_img + 1

    num_phases = km.cluster_centers_.shape[0]
    summary_line = f"Numero de fases (clusters) identificadas: {num_phases}"

    # 4) Guardar resultados principales
    phase_tsv = os.path.join(out_dir, f"{sample}_phase_mask.tsv")
    phase_png = os.path.join(out_dir, f"{sample}_phase_mask.png")
    stats_csv = os.path.join(out_dir, "label_to_phase_map.csv")
    summary_txt = os.path.join(out_dir, f"{sample}_summary.txt")

    print(f">> Guardando mapa de fases TSV: {phase_tsv}")
    save_phase_map_tsv(labels_img, phase_tsv)

    print(f">> Guardando mapa de fases PNG (colormap 'rainbow'): {phase_png}")
    save_phase_map_png(labels_img, phase_png)

    print(">> Guardando tabla de FRAC + ratios por fase (CSV/Excel)...")
    xlsx_path = save_label_stats_and_excel(km, stats_csv)
    if xlsx_path:
        print(f"   - CSV : {stats_csv}")
        print(f"   - XLSX: {xlsx_path}")
    else:
        print(f"   - CSV : {stats_csv}")
        print("   - XLSX: no generado (falta 'openpyxl')")

    # 5) Resumen detallado con definiciones y fases
    print(f">> Generando resumen de fases: {summary_txt}")
    summary_text = build_summary_text(km)
    with open(summary_txt, "w") as f:
        f.write(summary_text + "\n")

    print(f">> {summary_line}")
    print(">> Listo.")


if __name__ == "__main__":
    main()
