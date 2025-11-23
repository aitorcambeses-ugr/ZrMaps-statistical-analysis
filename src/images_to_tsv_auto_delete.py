#!/usr/bin/env python3
import os
import sys
import numpy as np
from PIL import Image

def infer_element_name(filename):
    """
    Obtiene el nombre del elemento a partir del nombre del archivo.
    Usa el texto después del último '_' como nombre de elemento.
    """
    name, _ = os.path.splitext(os.path.basename(filename))
    if "_" in name:
        return name.split("_")[-1]
    return name

def image_to_gray(path):
    """
    Convierte una imagen a matriz 2D en escala de grises estándar.
    """
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(float)
    gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
    return gray

def main():
    if len(sys.argv) < 2:
        print("USO: python3 images_to_tsv_auto_delete.py <carpeta>")
        sys.exit(1)

    root = os.path.abspath(os.getcwd())
    target = sys.argv[1]
    input_dir = os.path.abspath(target)

    if not os.path.isdir(input_dir):
        input_dir = os.path.join(root, target)

    if not os.path.isdir(input_dir):
        print(f"[ERROR] No existe la carpeta: {input_dir}")
        sys.exit(1)

    folder_name = os.path.basename(input_dir.rstrip("/"))
    output_dir = os.path.join(input_dir, f"{folder_name}_mapas_tsv")
    os.makedirs(output_dir, exist_ok=True)

    print(f">> Procesando carpeta: {input_dir}")
    print(f">> Prefijo: {folder_name}_")
    print(f">> Carpeta de salida: {output_dir}\n")

    exts = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    files = sorted(os.listdir(input_dir))
    images = [f for f in files if os.path.splitext(f)[1].lower() in exts]

    if not images:
        print("[AVISO] No hay imágenes válidas en esta carpeta.")
        sys.exit(0)

    for fname in images:
        in_path = os.path.join(input_dir, fname)
        element = infer_element_name(fname)
        out_name = f"{folder_name}_{element}.tsv"
        out_path = os.path.join(output_dir, out_name)

        try:
            print(f"[OK] {fname} → {out_name}")
            gray = image_to_gray(in_path)
            np.savetxt(out_path, gray, fmt="%.6f", delimiter="\t")
            os.remove(in_path)
            print(f"     [BORRADO] {fname}")
        except Exception as e:
            print(f"[ERROR] Fallo procesando {fname}: {e}")
            print("        → No se borra la imagen.\n")

if __name__ == "__main__":
    main()