# -*- coding: utf-8 -*-
"""Genera las figuras del Capítulo III a partir de los resultados obtenidos."""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SALIDA = Path(__file__).resolve().parent / "figuras"
SALIDA.mkdir(exist_ok=True)

GENEROS = ["Panstrongylus", "Rhodnius", "Triatoma"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 16,
    "axes.linewidth": 1.0,
    "figure.dpi": 300,
})

AZUL = LinearSegmentedColormap.from_list("azul", ["#ffffff", "#1f4e79"])
COMA = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))


def matriz_confusion(cm, archivo):
    cm = np.array(cm)
    total = cm.sum()

    fig, ax = plt.subplots(figsize=(8.0, 6.6))
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap=AZUL, vmin=0, vmax=1)

    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:,}".replace(",", "."),
                    ha="center", va="center", fontsize=24,
                    color="white" if norm[i, j] > 0.5 else "#1a1a1a",
                    fontweight="bold" if i == j else "normal")

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(GENEROS, style="italic", fontsize=17)
    ax.set_yticklabels(GENEROS, style="italic", fontsize=17, rotation=90, va="center")
    ax.set_xlabel("Género predicho", fontsize=19, labelpad=14)
    ax.set_ylabel("Género real", fontsize=19, labelpad=14)

    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="#9a9a9a", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0, pad=10)
    for s in ax.spines.values():
        s.set_edgecolor("#6a6a6a")

    exactitud = f"{np.trace(cm) / total:.4f}".replace(".", ",")
    ax.text(0.5, -0.20, f"n = {total:,}".replace(",", ".") + f"   ·   exactitud = {exactitud}",
            transform=ax.transAxes, ha="center", fontsize=15, color="#404040")

    fig.savefig(SALIDA / archivo, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ", archivo)


def curvas(hist1, hist2, metrica, etiqueta, archivo, leyenda):
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    epocas = np.arange(1, 11)
    tr = hist1["tr_" + metrica] + hist2["tr_" + metrica]
    va = hist1["va_" + metrica] + hist2["va_" + metrica]

    ax.axvspan(0.5, 5.5, color="#f2f2f2", zorder=0)
    ax.axvline(5.5, color="#8a8a8a", linewidth=1.2, linestyle="--", zorder=1)

    ax.plot(epocas, tr, "o-", color="#1f4e79", linewidth=2.6,
            markersize=8, label="Entrenamiento", zorder=3)
    ax.plot(epocas, va, "s--", color="#c1621f", linewidth=2.6,
            markersize=8, label="Validación", zorder=3)

    # Etiquetas de fase por encima del área del gráfico, para no chocar con la leyenda
    ax.text(0.25, 1.10, "Fase 1", transform=ax.transAxes,
            ha="center", fontsize=17, color="#404040")
    ax.text(0.25, 1.02, "base congelada", transform=ax.transAxes,
            ha="center", fontsize=14, color="#606060")
    ax.text(0.75, 1.10, "Fase 2", transform=ax.transAxes,
            ha="center", fontsize=17, color="#404040")
    ax.text(0.75, 1.02, "ajuste fino", transform=ax.transAxes,
            ha="center", fontsize=14, color="#606060")

    ymin, ymax = min(min(tr), min(va)), max(max(tr), max(va))
    margen = (ymax - ymin) * 0.10
    ax.set_xlabel("Época", fontsize=19, labelpad=12)
    ax.set_ylabel(etiqueta, fontsize=19, labelpad=12)
    ax.set_xticks(epocas)
    ax.tick_params(labelsize=16)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(ymin - margen, ymax + margen)
    ax.yaxis.set_major_formatter(COMA)
    ax.legend(frameon=False, loc=leyenda, fontsize=17)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)

    fig.savefig(SALIDA / archivo, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ", archivo)


print("Generando figuras...")

matriz_confusion([[1452, 5, 43], [1, 1491, 8], [3, 6, 1491]],
                 "FIGURA_4_matriz_modelo_A.png")

matriz_confusion([[125, 5, 18], [0, 36, 2], [18, 7, 200]],
                 "FIGURA_5_matriz_modelo_B.png")

matriz_confusion([[1679, 10, 51], [3, 1543, 10], [38, 23, 1889]],
                 "FIGURA_3_matriz_modelo_final.png")

FASE1 = {
    "tr_acc": [0.8109, 0.8693, 0.8794, 0.8838, 0.8832],
    "va_acc": [0.8910, 0.9024, 0.9007, 0.9056, 0.9079],
    "tr_loss": [0.4461, 0.3200, 0.3011, 0.2908, 0.2897],
    "va_loss": [0.2949, 0.2655, 0.2670, 0.2499, 0.2495],
}
FASE2 = {
    "tr_acc": [0.9197, 0.9462, 0.9611, 0.9697, 0.9759],
    "va_acc": [0.9447, 0.9581, 0.9628, 0.9724, 0.9743],
    "tr_loss": [0.2051, 0.1347, 0.0987, 0.0755, 0.0624],
    "va_loss": [0.1517, 0.1154, 0.0986, 0.0772, 0.0699],
}

curvas(FASE1, FASE2, "acc", "Exactitud", "FIGURA_1_curvas_exactitud.png", "lower right")
curvas(FASE1, FASE2, "loss", "Pérdida", "FIGURA_2_curvas_perdida.png", "upper right")

print(f"\nListo. {len(list(SALIDA.glob('*.png')))} figuras en: {SALIDA}")
