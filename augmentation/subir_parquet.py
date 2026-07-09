"""Genera el dataset augmentado y lo publica en Hugging Face en formato Parquet.

A diferencia de main.py (que sube imágenes como archivos sueltos), aquí se
construye un datasets.Dataset y se usa push_to_hub, que almacena los datos en
Parquet: unos pocos archivos grandes en lugar de 30.000 pequeños. El dataset
resultante se carga en segundos. Reutiliza la lógica de augment.py.
"""

import os
import sys

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")  # evita el chequeo de versión por red

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import Dataset, Features, ClassLabel
from datasets import Image as ImagenParquet

import config
import augment

REPO_PARQUET = config.HF_REPO_ID + "-parquet"


def main():
    config.validar_config()

    conteo = augment.contar_imagenes_por_clase()
    conteos = {clase: len(rutas) for clase, rutas in conteo.items()}
    plan = augment.calcular_augmentaciones(conteos, config.TARGET_TOTAL)
    clases = sorted(plan)
    label2id = {clase: i for i, clase in enumerate(clases)}
    total = sum(plan[c]["aug_total"] for c in clases)

    pipeline = augment.construir_pipeline()

    # Generador como closure: evita pasar `clases` por gen_kwargs (datasets lo
    # particiona por clase y rompería el cálculo de la etiqueta).
    def generar_ejemplos():
        for clase in clases:
            plan_clase = plan[clase]
            for indice, ruta in enumerate(conteo[clase]):
                n_aug = augment.augmentaciones_para_imagen(plan_clase, indice)
                if n_aug == 0:
                    continue
                imagen = augment.cargar_imagen(ruta)
                for _ in range(n_aug):
                    aug = augment.generar_augmentacion(imagen, pipeline)
                    jpeg = augment.imagen_a_bytes(aug)
                    yield {"image": {"bytes": jpeg, "path": None}, "label": label2id[clase]}

    print(f"Clases: {clases}")
    print(f"Generando {total} imágenes y publicando en {REPO_PARQUET}")

    features = Features({"image": ImagenParquet(), "label": ClassLabel(names=clases)})
    dataset = Dataset.from_generator(generar_ejemplos, features=features)

    print(f"Subiendo {dataset.num_rows} imágenes en Parquet...")
    dataset.push_to_hub(REPO_PARQUET, token=config.HF_TOKEN)
    print(f"Listo: https://huggingface.co/datasets/{REPO_PARQUET}")


if __name__ == "__main__":
    main()
