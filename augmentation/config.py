"""Configuración del sistema de data augmentation. Los valores se leen del archivo .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

RUTA_BASE = Path(__file__).resolve().parent
load_dotenv(RUTA_BASE / ".env")


def _leer_int(nombre, por_defecto):
    valor = os.getenv(nombre)
    if valor is None or valor.strip() == "":
        return por_defecto
    try:
        return int(valor)
    except ValueError as error:
        raise ValueError(
            f"La variable {nombre} debe ser un número entero (valor recibido: {valor!r})."
        ) from error


HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_REPO_ID = os.getenv("HF_REPO_ID", "tu-usuario/triatominos-augmentado").strip()
DATASET_PATH = os.getenv("DATASET_PATH", "/ruta/a/tu/dataset").strip()

TARGET_TOTAL = _leer_int("TARGET_TOTAL", 30_000)
TARGET_MIN = 20_000
TARGET_MAX = 40_000

IMG_SIZE = _leer_int("IMG_SIZE", 224)

TAMANO_LOTE_COMMIT = _leer_int("TAMANO_LOTE_COMMIT", 500)
REINTENTOS_SUBIDA = _leer_int("REINTENTOS_SUBIDA", 4)
CALIDAD_JPEG = _leer_int("CALIDAD_JPEG", 95)

EXTENSIONES_VALIDAS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

_PLACEHOLDERS = {
    "tu-usuario/triatominos-augmentado",
    "/ruta/a/tu/dataset",
    "your_token_here",
    "",
}


def validar_config():
    """Verifica que el token, el repositorio y la ruta del dataset estén definidos."""
    errores = []

    if HF_TOKEN in _PLACEHOLDERS:
        errores.append(
            "Falta HF_TOKEN. Crea un token con permiso de escritura en "
            "https://huggingface.co/settings/tokens y ponlo en el archivo .env"
        )

    if HF_REPO_ID in _PLACEHOLDERS or "/" not in HF_REPO_ID:
        errores.append("HF_REPO_ID inválido. Debe tener el formato 'usuario/nombre-dataset'.")

    ruta = Path(DATASET_PATH)
    if DATASET_PATH in _PLACEHOLDERS or not ruta.exists():
        errores.append(f"DATASET_PATH no existe o no está configurado: {DATASET_PATH!r}.")
    elif not ruta.is_dir():
        errores.append(f"DATASET_PATH debe ser una carpeta: {DATASET_PATH!r}")

    if errores:
        raise ValueError("Configuración incompleta:\n" + "\n".join(f"  - {e}" for e in errores))

    return True


def resumen_config():
    """Resumen de la configuración activa (sin exponer el token)."""
    token_estado = "definido" if HF_TOKEN not in _PLACEHOLDERS else "sin definir"
    return (
        "Configuración:\n"
        f"  Dataset : {DATASET_PATH}\n"
        f"  Repo    : {HF_REPO_ID}\n"
        f"  Objetivo: {TARGET_TOTAL:,} imágenes\n"
        f"  Tamaño  : {IMG_SIZE}x{IMG_SIZE}\n"
        f"  Lote    : {TAMANO_LOTE_COMMIT}\n"
        f"  Token   : {token_estado}"
    )
