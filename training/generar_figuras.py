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


def matriz_confusion(cm, archivo, etiquetas=GENEROS, eje="Género"):
    cm = np.array(cm)
    total = cm.sum()
    n = len(etiquetas)

    fig, ax = plt.subplots(figsize=(8.0, 6.6) if n == 3 else (9.4, 7.8))
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap=AZUL, vmin=0, vmax=1)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]:,}".replace(",", "."),
                    ha="center", va="center", fontsize=24 if n == 3 else 22,
                    color="white" if norm[i, j] > 0.5 else "#1a1a1a",
                    fontweight="bold" if i == j else "normal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(etiquetas, fontsize=17)
    ax.set_yticklabels(etiquetas, fontsize=17, rotation=90, va="center")
    # Los géneros van en cursiva; "Otro" no es un nombre científico.
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_style("normal" if t.get_text() == "Otro" else "italic")
    ax.set_xlabel(f"{eje} predicho" if eje == "Género" else f"{eje} predicha", fontsize=19, labelpad=14)
    ax.set_ylabel(f"{eje} real", fontsize=19, labelpad=14)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
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


def matrices_especie(paneles, archivo):
    """Una matriz 2 × 2 por capa de especie (validación cruzada): otra especie vs. especie objetivo."""
    fig, ejes = plt.subplots(1, len(paneles), figsize=(6.4 * len(paneles), 6.6))
    for k, (ax, (especie, cm)) in enumerate(zip(ejes, paneles)):
        cm = np.array(cm)
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap=AZUL, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}".replace(",", "."), ha="center", va="center", fontsize=22,
                        color="white" if norm[i, j] > 0.5 else "#1a1a1a",
                        fontweight="bold" if i == j else "normal")
        etiquetas = ["Otra especie", especie]
        ax.set_xticks(range(2))
        ax.set_yticks(range(2))
        ax.set_xticklabels(etiquetas, fontsize=16)
        ax.set_yticklabels(etiquetas, fontsize=16, rotation=90, va="center")
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_style("normal" if t.get_text() == "Otra especie" else "italic")
        ax.set_title(especie, fontsize=20, style="italic", pad=14)
        ax.set_xlabel("Clase predicha", fontsize=17, labelpad=12)
        if k == 0:
            ax.set_ylabel("Clase real", fontsize=17, labelpad=12)
        ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
        ax.grid(which="minor", color="#9a9a9a", linewidth=1.0)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(which="major", length=0, pad=10)
        for s in ax.spines.values():
            s.set_edgecolor("#6a6a6a")
        exh = f"{cm[1, 1] / cm[1].sum():.4f}".replace(".", ",")
        esp = f"{cm[0, 0] / cm[0].sum():.4f}".replace(".", ",")
        ax.text(0.5, -0.24, f"exhaustividad = {exh}   ·   especificidad = {esp}",
                transform=ax.transAxes, ha="center", fontsize=14, color="#404040")
    fig.tight_layout(w_pad=3)
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

# Modelo final de cuatro clases (notebook "Clasificador de triatominos — MobileNetV3 FINAL4").
CLASES = ["Panstrongylus", "Rhodnius", "Triatoma", "Otro"]
matriz_confusion([[1699, 4, 46, 3], [2, 1544, 13, 1], [37, 18, 1885, 10], [4, 4, 20, 2142]],
                 "FIGURA_8_matriz_modelo_final_4_clases.png", etiquetas=CLASES, eje="Clase")

FASE1_4 = {
    "tr_acc": [0.8390, 0.8851, 0.8899, 0.8909, 0.8925],
    "va_acc": [0.9046, 0.9199, 0.9206, 0.9210, 0.9245],
    "tr_loss": [0.4174, 0.3043, 0.2902, 0.2911, 0.2888],
    "va_loss": [0.2636, 0.2292, 0.2261, 0.2251, 0.2145],
}
FASE2_4 = {
    "tr_acc": [0.9266, 0.9560, 0.9673, 0.9750, 0.9792],
    "va_acc": [0.9553, 0.9672, 0.9748, 0.9760, 0.9782],
    "tr_loss": [0.1983, 0.1248, 0.0921, 0.0711, 0.0560],
    "va_loss": [0.1283, 0.0946, 0.0757, 0.0700, 0.0580],
}
curvas(FASE1_4, FASE2_4, "acc", "Exactitud", "FIGURA_6_curvas_exactitud_4_clases.png", "lower right")
curvas(FASE1_4, FASE2_4, "loss", "Pérdida", "FIGURA_7_curvas_perdida_4_clases.png", "upper right")

# Capas de especie sobre el modelo final (notebook cabezales_especie, validación cruzada agrupada por ejemplar).
matrices_especie([("P. geniculatus", [[1188, 130], [49, 1056]]),
                  ("R. prolixus", [[1415, 119], [8, 72]]),
                  ("T. maculata", [[4240, 463], [15, 259]])],
                 "FIGURA_9_matrices_capas_especie.png")

print(f"\nListo. {len(list(SALIDA.glob('*.png')))} figuras en: {SALIDA}")
