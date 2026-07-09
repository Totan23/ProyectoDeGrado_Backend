"""Data augmentation en memoria para el dataset de triatominos."""

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
import albumentations as A

import config


def contar_imagenes_por_clase(dataset_path=None):
    """Agrupa las imágenes del dataset por clase (carpeta). Devuelve {clase: [rutas]}."""
    raiz = Path(dataset_path or config.DATASET_PATH)
    if not raiz.is_dir():
        raise FileNotFoundError(f"No existe la carpeta del dataset: {raiz}")

    clases = {}
    for archivo in sorted(raiz.rglob("*")):
        if archivo.is_file() and archivo.suffix.lower() in config.EXTENSIONES_VALIDAS:
            carpeta_rel = archivo.parent.relative_to(raiz).as_posix()
            if carpeta_rel in (".", ""):
                carpeta_rel = "raiz"
            clases.setdefault(carpeta_rel, []).append(archivo)

    if not clases:
        raise ValueError(
            f"No se encontraron imágenes ({', '.join(config.EXTENSIONES_VALIDAS)}) en {raiz}"
        )
    return clases


def calcular_augmentaciones(conteo_por_clase, objetivo_total=None):
    """Reparte objetivo_total imágenes de forma balanceada entre las clases y
    devuelve, por clase, cuántas augmentaciones generar por imagen."""
    objetivo_total = objetivo_total if objetivo_total is not None else config.TARGET_TOTAL

    conteos = {
        clase: (valor if isinstance(valor, int) else len(valor))
        for clase, valor in conteo_por_clase.items()
    }
    clases = sorted(clase for clase, n in conteos.items() if n > 0)
    num_clases = len(clases)
    if num_clases == 0:
        raise ValueError("No hay clases con imágenes para augmentar.")

    base_por_clase = objetivo_total // num_clases
    resto_clases = objetivo_total % num_clases

    plan = {}
    for indice, clase in enumerate(clases):
        n = conteos[clase]
        objetivo_clase = base_por_clase + (1 if indice < resto_clases else 0)
        base_por_imagen = objetivo_clase // n
        imagenes_con_extra = objetivo_clase % n

        plan[clase] = {
            "imagenes": n,
            "objetivo": objetivo_clase,
            "base_por_imagen": base_por_imagen,
            "imagenes_con_extra": imagenes_con_extra,
            "aug_total": objetivo_clase,
        }
    return plan


def augmentaciones_para_imagen(plan_clase, indice_imagen):
    """Número de augmentaciones para una imagen según el plan de su clase."""
    extra = 1 if indice_imagen < plan_clase["imagenes_con_extra"] else 0
    return plan_clase["base_por_imagen"] + extra


def _crear_con_fallback(*constructores):
    """Construye una transformación probando varias firmas (compatibilidad entre
    albumentations 1.x y 2.x)."""
    ultimo_error = None
    for constructor in constructores:
        try:
            return constructor()
        except TypeError as error:
            ultimo_error = error
    raise ultimo_error


def construir_pipeline(img_size=None):
    """Pipeline de transformaciones con salida estandarizada a img_size x img_size."""
    lado = img_size or config.IMG_SIZE

    recorte_redimensionado = _crear_con_fallback(
        lambda: A.RandomResizedCrop(size=(lado, lado), scale=(0.6, 1.0),
                                    ratio=(0.75, 1.3333), p=0.5),
        lambda: A.RandomResizedCrop(height=lado, width=lado, scale=(0.6, 1.0),
                                    ratio=(0.75, 1.3333), p=0.5),
    )
    ruido_gaussiano = _crear_con_fallback(
        lambda: A.GaussNoise(std_range=(0.04, 0.12), p=0.3),
        lambda: A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    )

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        recorte_redimensionado,
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.4),
        ruido_gaussiano,
        A.Blur(blur_limit=3, p=0.2),
        A.Resize(height=lado, width=lado),
    ])


_pipeline_cache = None


def _pipeline_por_defecto():
    global _pipeline_cache
    if _pipeline_cache is None:
        _pipeline_cache = construir_pipeline(config.IMG_SIZE)
    return _pipeline_cache


def cargar_imagen(ruta):
    """Carga una imagen como array numpy RGB, respetando la orientación EXIF."""
    with Image.open(ruta) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        return np.array(img)


def generar_augmentacion(imagen, pipeline=None):
    """Aplica el pipeline a una imagen (array numpy RGB) y devuelve el resultado."""
    pipeline = pipeline or _pipeline_por_defecto()
    return pipeline(image=imagen)["image"]


def imagen_a_bytes(imagen_np, calidad=None):
    """Codifica una imagen (array numpy RGB) a bytes JPEG."""
    calidad = calidad if calidad is not None else config.CALIDAD_JPEG
    img = Image.fromarray(np.asarray(imagen_np, dtype=np.uint8))
    if img.mode != "RGB":
        img = img.convert("RGB")
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=calidad)
    return buffer.getvalue()
