"""Corrige las etiquetas del dataset Parquet ya publicado, sin regenerar imágenes.

Las imágenes se subieron correctas y en orden (bloques por clase), pero un bug en
la subida dejó todas las etiquetas en 0. Aquí se reasignan por posición, según el
mismo orden y conteo con que se generaron, y se vuelve a publicar.
"""

import os
import sys

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collections import Counter

from datasets import load_dataset, ClassLabel

import config
import augment

REPO_PARQUET = config.HF_REPO_ID + "-parquet"


def main():
    # Reconstruye el plan para saber el orden y el nº de imágenes por clase.
    conteo = augment.contar_imagenes_por_clase()
    conteos = {clase: len(rutas) for clase, rutas in conteo.items()}
    plan = augment.calcular_augmentaciones(conteos, config.TARGET_TOTAL)
    clases = sorted(plan)
    label2id = {clase: i for i, clase in enumerate(clases)}

    # Etiquetas por posición: los ejemplos se generaron en bloques por clase, en orden.
    etiquetas = []
    for clase in clases:
        etiquetas += [label2id[clase]] * plan[clase]["aug_total"]

    print("Descargando el dataset (1 archivo, rápido)...")
    ds = load_dataset(REPO_PARQUET, split="train")
    print(f"Filas en el dataset: {ds.num_rows} | etiquetas según el plan: {len(etiquetas)}")

    if ds.num_rows != len(etiquetas):
        raise SystemExit(
            f"El nº de filas ({ds.num_rows}) no coincide con el plan ({len(etiquetas)}). "
            "No se re-etiqueta para no arriesgar las clases; avísame."
        )

    ds = ds.remove_columns("label").add_column("label", etiquetas)
    ds = ds.cast_column("label", ClassLabel(names=clases))
    print("Etiquetas corregidas:", dict(Counter(ds["label"])))

    print("Subiendo la corrección...")
    ds.push_to_hub(REPO_PARQUET, token=config.HF_TOKEN)
    print(f"Listo: https://huggingface.co/datasets/{REPO_PARQUET}")


if __name__ == "__main__":
    main()
