
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


# ============================================================
# DIABETES REGRESSION AI - SHAP EXPLAINABILITY
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


def main():
    print("=" * 70)
    print("DIABETES REGRESSION AI - SHAP EXPLAINABILITY")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Caricamento del dataset
    # --------------------------------------------------------
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.data
    y = diabetes.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print(f"\nOsservazioni totali: {len(X)}")
    print(f"Training set:        {len(X_train)}")
    print(f"Test set:            {len(X_test)}")

    # --------------------------------------------------------
    # 2. Ricostruzione del Gradient Boosting ottimizzato
    # Parametri ottenuti dalla GridSearchCV precedente
    # --------------------------------------------------------
    model = GradientBoostingRegressor(
        learning_rate=0.1,
        max_depth=1,
        min_samples_leaf=3,
        n_estimators=300,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    print("\nModello: GradientBoostingRegressor")
    print("Parametri utilizzati:")
    print(model.get_params())

    # --------------------------------------------------------
    # 3. Predizioni e valori SHAP
    # --------------------------------------------------------
    print("\nCalcolo dei valori SHAP...")

    # TreeExplainer è specifico per i modelli ad alberi
    explainer = shap.TreeExplainer(model)

    # Spieghiamo le osservazioni del test set
    shap_values = explainer(X_test)

    values = np.asarray(shap_values.values)
    base_value = float(np.asarray(shap_values.base_values).mean())

    predictions = model.predict(X_test)

    # Verifica della coerenza:
    # previsione = valore base + somma dei contributi SHAP
    reconstructed = base_value + values.sum(axis=1)
    max_difference = float(
        np.max(np.abs(predictions - reconstructed))
    )

    print(f"Valore base medio: {base_value:.4f}")
    print(f"Massimo errore di ricostruzione: {max_difference:.8f}")

    # --------------------------------------------------------
    # 4. Importanza globale SHAP
    # Media del valore assoluto SHAP per ogni feature
    # --------------------------------------------------------
    importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Mean_Absolute_SHAP": np.abs(values).mean(axis=0)
    })

    importance = importance.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    ).reset_index(drop=True)

    importance_path = METRICS_DIR / "shap_feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    print("\nIMPORTANZA GLOBALE SHAP")
    print("-" * 70)
    print(
        importance.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    # --------------------------------------------------------
    # 5. Grafico SHAP summary: importanza e distribuzione
    # --------------------------------------------------------
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_test,
        show=False,
        plot_size=None
    )
    plt.title("SHAP Summary - Gradient Boosting")
    plt.tight_layout()

    summary_path = FIGURES_DIR / "shap_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------
    # 6. Grafico a barre dell'importanza globale
    # --------------------------------------------------------
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=10, show=False)
    plt.title("SHAP Feature Importance - Gradient Boosting")
    plt.tight_layout()

    bar_path = FIGURES_DIR / "shap_importance_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------
    # 7. Spiegazione delle singole previsioni
    # --------------------------------------------------------
    prediction_results = X_test.copy().reset_index(drop=True)

    prediction_results["Actual"] = y_test.reset_index(drop=True)
    prediction_results["Predicted"] = predictions
    prediction_results["Absolute_Error"] = np.abs(
        prediction_results["Actual"] - prediction_results["Predicted"]
    )

    prediction_results["SHAP_Base_Value"] = base_value

    # Salva le spiegazioni di ogni feature per ogni osservazione
    shap_df = pd.DataFrame(
        values,
        columns=[f"SHAP_{column}" for column in X_test.columns]
    ).reset_index(drop=True)

    prediction_results = pd.concat(
        [prediction_results, shap_df],
        axis=1
    )

    predictions_path = METRICS_DIR / "shap_test_predictions.csv"
    prediction_results.to_csv(predictions_path, index=False)

    # --------------------------------------------------------
    # 8. Osservazione con errore assoluto maggiore
    # --------------------------------------------------------
    worst_index = int(
        np.argmax(np.abs(y_test.to_numpy() - predictions))
    )

    print("\nOSSERVAZIONE CON ERRORE ASSOLUTO MAGGIORE")
    print("-" * 70)
    print(f"Valore reale:     {y_test.iloc[worst_index]:.3f}")
    print(f"Previsione:       {predictions[worst_index]:.3f}")
    print(
        "Errore assoluto: "
        f"{abs(y_test.iloc[worst_index] - predictions[worst_index]):.3f}"
    )

    # Waterfall plot per la singola osservazione
    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(
        shap_values[worst_index],
        max_display=10,
        show=False
    )

    waterfall_path = FIGURES_DIR / "shap_worst_prediction.png"
    plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------
    # 9. Salvataggio metadati
    # --------------------------------------------------------
    metadata = {
        "model": "GradientBoostingRegressor",
        "parameters": {
            "learning_rate": 0.1,
            "max_depth": 1,
            "min_samples_leaf": 3,
            "n_estimators": 300,
            "random_state": RANDOM_STATE
        },
        "explained_dataset": "scikit-learn diabetes",
        "explained_split": "test set",
        "n_test_observations": len(X_test),
        "mean_base_value": base_value,
        "max_reconstruction_difference": max_difference,
        "note": (
            "SHAP describes model contributions, not causal effects "
            "or clinical explanations."
        )
    }

    metadata_path = METRICS_DIR / "shap_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    print("\nFILE GENERATI")
    print("-" * 70)
    print(f"Importanza SHAP:       {importance_path}")
    print(f"Predizioni e SHAP:     {predictions_path}")
    print(f"Metadati:              {metadata_path}")
    print(f"Grafico summary:       {summary_path}")
    print(f"Grafico a barre:       {bar_path}")
    print(f"Waterfall individuale: {waterfall_path}")

    print("\nANALISI SHAP COMPLETATA")


if __name__ == "__main__":
    main()
