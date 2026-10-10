
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes


# ============================================================
# DIABETES REGRESSION AI - RIDGE EXPLAINABILITY
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_regression_model.joblib"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("DIABETES REGRESSION AI - RIDGE EXPLAINABILITY")
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modello non trovato: {MODEL_PATH}\n"
            "Esegui prima src/optimize_models.py."
        )

    # Caricamento del modello selezionato
    model = joblib.load(MODEL_PATH)

    # Caricamento dei dati e dei nomi delle variabili
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.data
    feature_names = X.columns.tolist()

    # Recupera il regressore finale, sia che sia dentro una Pipeline
    # sia che il file contenga direttamente un modello Ridge.
    if hasattr(model, "named_steps"):
        regressor = model.named_steps["model"]
    else:
        regressor = model

    if not hasattr(regressor, "coef_"):
        raise TypeError(
            "Il modello caricato non espone i coefficienti coef_."
        )

    coefficients = np.asarray(regressor.coef_).ravel()

    if len(coefficients) != len(feature_names):
        raise ValueError(
            "Numero di coefficienti diverso dal numero di variabili: "
            f"{len(coefficients)} contro {len(feature_names)}."
        )

    # Tabella dei coefficienti
    results = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients,
        "Absolute_Coefficient": np.abs(coefficients),
    })

    results["Direction"] = np.where(
        results["Coefficient"] > 0,
        "Positiva",
        np.where(
            results["Coefficient"] < 0,
            "Negativa",
            "Nulla"
        )
    )

    results = results.sort_values(
        "Absolute_Coefficient",
        ascending=False
    ).reset_index(drop=True)

    output_csv = METRICS_DIR / "ridge_coefficients.csv"
    results.to_csv(output_csv, index=False)

    # Stampa a terminale
    print("\nCOEFFICIENTI RIDGE ORDINATI PER VALORE ASSOLUTO")
    print("-" * 70)
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nCOME LEGGERE I COEFFICIENTI")
    print("-" * 70)
    print(
        "Coefficiente positivo: all'aumentare della variabile, "
        "la previsione tende ad aumentare, a parita delle altre variabili."
    )
    print(
        "Coefficiente negativo: all'aumentare della variabile, "
        "la previsione tende a diminuire, a parita delle altre variabili."
    )
    print(
        "Il valore assoluto permette un confronto approssimativo "
        "dell'influenza lineare, ma non dimostra un rapporto causale."
    )

    # Grafico dei coefficienti con segno
    signed = results.sort_values("Coefficient")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [
        "steelblue" if value >= 0 else "indianred"
        for value in signed["Coefficient"]
    ]

    ax.barh(
        signed["Feature"],
        signed["Coefficient"],
        color=colors
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Ridge: coefficienti delle variabili")
    ax.set_xlabel("Coefficiente")
    ax.set_ylabel("Variabile")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    signed_plot = FIGURES_DIR / "ridge_coefficients_signed.png"
    fig.savefig(signed_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Grafico dell'importanza relativa: valore assoluto
    importance = results.sort_values("Absolute_Coefficient")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        importance["Feature"],
        importance["Absolute_Coefficient"],
        color="slateblue"
    )
    ax.set_title("Ridge: importanza relativa dei coefficienti")
    ax.set_xlabel("Valore assoluto del coefficiente")
    ax.set_ylabel("Variabile")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    importance_plot = FIGURES_DIR / "ridge_feature_importance.png"
    fig.savefig(importance_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nFILE GENERATI")
    print("-" * 70)
    print(f"Tabella CSV: {output_csv}")
    print(f"Coefficienti con segno: {signed_plot}")
    print(f"Importanza relativa: {importance_plot}")

    print("\nANALISI COMPLETATA")


if __name__ == "__main__":
    main()
