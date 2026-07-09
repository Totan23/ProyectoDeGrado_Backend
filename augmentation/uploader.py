"""Subida de imágenes a Hugging Face Datasets, agrupadas en lotes, vía huggingface_hub."""

import time
from collections import defaultdict

from huggingface_hub import HfApi, CommitOperationAdd
from huggingface_hub.utils import HfHubHTTPError

try:
    from requests.exceptions import RequestException
except ImportError:
    RequestException = Exception

import config


_ERRORES_REINTENTABLES = (HfHubHTTPError, RequestException, ConnectionError, TimeoutError)


class SubidorHF:
    """Acumula imágenes y las sube a Hugging Face en lotes (un commit por lote)."""

    def __init__(self, repo_id=None, token=None, tamano_lote=None, reintentos=None):
        self.repo_id = repo_id or config.HF_REPO_ID
        self.token = token or config.HF_TOKEN
        self.tamano_lote = tamano_lote or config.TAMANO_LOTE_COMMIT
        self.reintentos = reintentos or config.REINTENTOS_SUBIDA
        self.api = HfApi(token=self.token)

        self._operaciones = []
        self._subidas_por_clase = defaultdict(int)
        self._lotes_subidos = 0

    def asegurar_repo(self):
        """Crea el repositorio de tipo dataset si no existe."""
        self.api.create_repo(repo_id=self.repo_id, repo_type="dataset", exist_ok=True)
        return f"https://huggingface.co/datasets/{self.repo_id}"

    def agregar_imagen(self, clase, nombre_archivo, datos_bytes):
        """Encola una imagen bajo la carpeta de su clase y sube el lote si está lleno."""
        ruta_en_repo = f"{clase}/{nombre_archivo}"
        self._operaciones.append(
            CommitOperationAdd(path_in_repo=ruta_en_repo, path_or_fileobj=datos_bytes)
        )
        self._subidas_por_clase[clase] += 1

        if len(self._operaciones) >= self.tamano_lote:
            self._subir_lote()

    def _subir_lote(self):
        """Sube el lote actual con reintentos y backoff exponencial."""
        if not self._operaciones:
            return

        operaciones = self._operaciones
        n = len(operaciones)
        mensaje = f"Augmentación de triatominos: lote {self._lotes_subidos + 1} ({n} imágenes)"

        ultimo_error = None
        for intento in range(1, self.reintentos + 1):
            try:
                self.api.create_commit(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    operations=operaciones,
                    commit_message=mensaje,
                )
                self._operaciones = []
                self._lotes_subidos += 1
                return
            except _ERRORES_REINTENTABLES as error:
                ultimo_error = error
                espera = min(2 ** intento, 30)
                print(f"  Fallo al subir el lote (intento {intento}/{self.reintentos}): "
                      f"{error}. Reintentando en {espera}s.")
                time.sleep(espera)

        raise RuntimeError(
            f"No se pudo subir el lote tras {self.reintentos} intentos. Último error: {ultimo_error}"
        )

    def finalizar(self):
        """Sube el lote pendiente y devuelve el conteo de imágenes por clase."""
        self._subir_lote()
        return dict(self._subidas_por_clase)

    @property
    def total_subidas(self):
        return sum(self._subidas_por_clase.values())
