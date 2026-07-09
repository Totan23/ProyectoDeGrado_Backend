"""Orquestador del data augmentation: escanea, planifica, genera y sube a Hugging Face."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tqdm import tqdm

import config
import augment
from uploader import SubidorHF


def mostrar_conteo(conteo):
    print("\nConteo de imágenes por clase (dataset original)")
    print("-" * 56)
    total = 0
    for clase in sorted(conteo):
        n = len(conteo[clase])
        total += n
        print(f"  {clase:<44} {n:>6}")
    print("-" * 56)
    print(f"  {'Total':<44} {total:>6}")
    print(f"  Clases: {len(conteo)}")


def mostrar_plan(plan):
    print("\nPlan de augmentación balanceada")
    print("-" * 64)
    print(f"  {'Clase':<34}{'Orig.':>7}{'Aug/img':>10}{'Total':>11}")
    print("-" * 64)
    total_aug = 0
    for clase in sorted(plan):
        p = plan[clase]
        total_aug += p["aug_total"]
        if p["imagenes_con_extra"]:
            aug_img = f"{p['base_por_imagen']}-{p['base_por_imagen'] + 1}"
        else:
            aug_img = str(p["base_por_imagen"])
        print(f"  {clase:<34}{p['imagenes']:>7}{aug_img:>10}{p['aug_total']:>11}")
    print("-" * 64)
    print(f"  {'Total a generar':<51}{total_aug:>11}")
    return total_aug


def confirmar(total_aug):
    print(f"\nSe generarán y subirán ~{total_aug:,} imágenes a:")
    print(f"  https://huggingface.co/datasets/{config.HF_REPO_ID}")
    while True:
        respuesta = input("¿Continuar? (y/n): ").strip().lower()
        if respuesta in ("y", "yes", "s", "si", "sí"):
            return True
        if respuesta in ("n", "no"):
            return False
        print("Responde 'y' o 'n'.")


def ejecutar(conteo, plan, total_aug):
    subidor = SubidorHF()
    print("\nPreparando el repositorio en Hugging Face...")
    url = subidor.asegurar_repo()
    print(f"Repositorio: {url}\n")

    barra = tqdm(total=total_aug, unit="img", desc="Augmentando y subiendo")
    try:
        for clase in sorted(plan):
            plan_clase = plan[clase]
            rutas = conteo[clase]
            contador = 0

            for indice_imagen, ruta in enumerate(rutas):
                n_aug = augment.augmentaciones_para_imagen(plan_clase, indice_imagen)
                if n_aug == 0:
                    continue

                try:
                    imagen = augment.cargar_imagen(ruta)
                except Exception as error:
                    barra.write(f"  No se pudo leer {ruta}: {error}. Se omiten {n_aug} augmentaciones.")
                    barra.update(n_aug)
                    continue

                for _ in range(n_aug):
                    contador += 1
                    aug = augment.generar_augmentacion(imagen)
                    datos = augment.imagen_a_bytes(aug)
                    nombre = f"aug_{contador:06d}.jpg"
                    subidor.agregar_imagen(clase, nombre, datos)
                    barra.update(1)
    except KeyboardInterrupt:
        barra.close()
        print("\n\nInterrumpido. Subiendo lo que ya está en cola...")
        reporte = subidor.finalizar()
        reportar(reporte, parcial=True)
        sys.exit(1)

    barra.close()
    print("\nSubiendo el último lote...")
    return subidor.finalizar()


def reportar(reporte, parcial=False):
    titulo = "Reporte parcial" if parcial else "Reporte final"
    print(f"\n{titulo} — imágenes subidas por clase")
    print("-" * 56)
    total = 0
    for clase in sorted(reporte):
        n = reporte[clase]
        total += n
        print(f"  {clase:<44} {n:>6}")
    print("-" * 56)
    print(f"  {'Total':<44} {total:>6}")
    print(f"\nDataset: https://huggingface.co/datasets/{config.HF_REPO_ID}")


def main():
    print(config.resumen_config())

    try:
        config.validar_config()
    except ValueError as error:
        print(f"\n{error}\n")
        sys.exit(1)

    print("\nEscaneando el dataset...")
    conteo = augment.contar_imagenes_por_clase()
    mostrar_conteo(conteo)

    conteos_int = {clase: len(rutas) for clase, rutas in conteo.items()}
    plan = augment.calcular_augmentaciones(conteos_int, config.TARGET_TOTAL)
    total_aug = mostrar_plan(plan)

    if not (config.TARGET_MIN <= config.TARGET_TOTAL <= config.TARGET_MAX):
        print(f"\nAviso: TARGET_TOTAL={config.TARGET_TOTAL:,} está fuera del rango "
              f"recomendado ({config.TARGET_MIN:,}-{config.TARGET_MAX:,}).")

    if not confirmar(total_aug):
        print("Cancelado.")
        sys.exit(0)

    reporte = ejecutar(conteo, plan, total_aug)
    reportar(reporte)


if __name__ == "__main__":
    main()
