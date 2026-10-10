
"""
DIABETES REGRESSION AI
Fase 2: prima baseline di regressione.

Il modello impara dai dati di training e viene valutato
su osservazioni non utilizzate durante l'addestramento.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


# ============================================================
# 1. PERCORSI
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CARICAMENTO DATI
# ============================================================

def load_data():
    """Carica caratteristiche e target."""

    dataset = load_diabetes(as_frame=True)

    X = dataset.data
    y = dataset.target

    return X, y


# ============================================================
# 3. SUDDIVISIONE TRAINING / TEST
# ============================================================

def split_data(X, y):
    """Crea una suddivisione riproducibile dei dati."""

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )


# ============================================================
# 4. METRICHE DI VALUTAZIONE
# ============================================================

def calculate_metrics(y_true, y_pred):
    """Calcola metriche standard per la regressione."""

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


# ============================================================
# 5. ADDESTRAMENTO E CONFRONTO
# ============================================================

def train_models(X_train, X_test, y_train, y_test):
    """Confronta baseline ingenua e regressione lineare."""

    models = {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
    }

    results = []
    predictions = {}

    for name, model in models.items():

        # Addestramento esclusivamente sul training set
        model.fit(X_train, y_train)

        # Previsioni sulle osservazioni di test
        y_pred = model.predict(X_test)

        metrics = calculate_metrics(y_test, y_pred)

        results.append({
            "Model": name,
            **metrics,
        })

        predictions[name] = y_pred

        print(f"\nMODELLO: {name}")
        print("-" * 50)
        print(f"MAE:  {metrics['MAE']:.3f}")
        print(f"RMSE: {metrics['RMSE']:.3f}")
        print(f"R²:   {metrics['R2']:.3f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        METRICS_DIR / "baseline_metrics.csv",
        index=False,
    )

    predictions_df = pd.DataFrame({
        "Actual": y_test.to_numpy(),
        "DummyRegressor": predictions["DummyRegressor"],
        "LinearRegression": predictions["LinearRegression"],
    })

    predictions_df.to_csv(
        METRICS_DIR / "baseline_predictions.csv",
        index=False,
    )

    return results_df, predictions_df


# ============================================================
# 6. GRAFICO PREVISTO VS OSSERVATO
# ============================================================

def plot_predictions(predictions_df):
    """Confronta valori osservati e previsti."""

    y_true = predictions_df["Actual"]
    y_pred = predictions_df["LinearRegression"]

    lower = min(y_true.min(), y_pred.min())
    upper = max(y_true.max(), y_pred.max())

    plt.figure(figsize=(8, 7))

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.75,
        edgecolors="black",
    )

    # Linea ideale: previsione uguale al valore osservato
    plt.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        label="Previsione ideale",
    )

    plt.xlabel("Valore osservato")
    plt.ylabel("Valore previsto")
    plt.title("Linear Regression: valori osservati vs previsti")
    plt.legend()
    plt.tight_layout()

    output_path = FIGURES_DIR / "baseline_predictions.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nGrafico salvato: {output_path}")


# ============================================================
# 7. PROGRAMMA PRINCIPALE
# ============================================================

def main():

    print("=" * 60)
    print("DIABETES REGRESSION AI - BASELINE")
    print("=" * 60)

    X, y = load_data()

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\nDIMENSIONI DEI DATI")
    print("-" * 50)
    print(f"Training set: {len(X_train)} osservazioni")
    print(f"Test set:     {len(X_test)} osservazioni")
    print(f"Caratteristiche: {X_train.shape[1]}")

    results_df, predictions_df = train_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\nCONFRONTO DEI MODELLI")
    print("-" * 50)
    print(results_df.round(3).to_string(index=False))

    plot_predictions(predictions_df)

    print("\nRISULTATI SALVATI")
    print(f"Metriche: {METRICS_DIR}")
    print(f"Previsioni: {METRICS_DIR}")
    print(f"Grafici: {FIGURES_DIR}")

    print("\nBASELINE COMPLETATA")


if __name__ == "__main__":
    main()
