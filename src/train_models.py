
"""
DIABETES REGRESSION AI
Fase 3: confronto tra modelli di Machine Learning.

Modelli:
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

La selezione usa cross-validation sul training set.
Il test set viene usato per la valutazione finale.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.datasets import load_diabetes
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. CONFIGURAZIONE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5


# ============================================================
# 2. CARICAMENTO E SUDDIVISIONE
# ============================================================

def load_and_split_data():
    """Carica i dati e crea training e test set."""

    dataset = load_diabetes(as_frame=True)

    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# 3. DEFINIZIONE DEI MODELLI
# ============================================================

def create_models():
    """Restituisce i quattro modelli da confrontare."""

    return {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),

        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),

        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            loss="squared_error",
            random_state=RANDOM_STATE,
        ),
    }


# ============================================================
# 4. CROSS-VALIDATION SUL TRAINING SET
# ============================================================

def evaluate_cross_validation(models, X_train, y_train):
    """
    Valuta ogni modello con 5 fold sul solo training set.

    Il MAE viene riportato come valore positivo per facilitarne
    l'interpretazione: valori più bassi sono migliori.
    """

    cv = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    results = []

    print("\nCROSS-VALIDATION SUL TRAINING SET")
    print("=" * 70)

    for name, model in models.items():

        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )

        mae_scores = -scores["test_mae"]
        rmse_scores = -scores["test_rmse"]
        r2_scores = scores["test_r2"]

        row = {
            "Model": name,
            "CV_MAE_mean": mae_scores.mean(),
            "CV_MAE_std": mae_scores.std(),
            "CV_RMSE_mean": rmse_scores.mean(),
            "CV_RMSE_std": rmse_scores.std(),
            "CV_R2_mean": r2_scores.mean(),
            "CV_R2_std": r2_scores.std(),
        }

        results.append(row)

        print(f"\n{name}")
        print("-" * 50)
        print(
            f"MAE:  {row['CV_MAE_mean']:.3f} "
            f"+/- {row['CV_MAE_std']:.3f}"
        )
        print(
            f"RMSE: {row['CV_RMSE_mean']:.3f} "
            f"+/- {row['CV_RMSE_std']:.3f}"
        )
        print(
            f"R²:   {row['CV_R2_mean']:.3f} "
            f"+/- {row['CV_R2_std']:.3f}"
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        METRICS_DIR / "cross_validation_metrics.csv",
        index=False,
    )

    return results_df


# ============================================================
# 5. ADDESTRAMENTO FINALE E TEST
# ============================================================

def evaluate_test_set(
    models,
    cv_results,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Seleziona il modello con MAE medio CV più basso,
    poi valuta tutti i modelli sul test set per confronto.
    """

    best_name = (
        cv_results
        .sort_values("CV_MAE_mean")
        .iloc[0]["Model"]
    )

    print("\nMODELLO SELEZIONATO DALLA CROSS-VALIDATION")
    print("=" * 70)
    print(f"Selezione basata sul MAE medio CV: {best_name}")

    test_results = []
    predictions = pd.DataFrame({
        "Actual": y_test.to_numpy()
    })

    fitted_models = {}

    for name, model in models.items():

        fitted_model = clone(model)
        fitted_model.fit(X_train, y_train)

        y_pred = fitted_model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        test_results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Selected_by_CV": name == best_name,
        })

        predictions[name] = y_pred
        fitted_models[name] = fitted_model

        print(f"\n{name}")
        print("-" * 50)
        print(f"MAE:  {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R²:   {r2:.3f}")

    test_results_df = pd.DataFrame(test_results)

    test_results_df.to_csv(
        METRICS_DIR / "model_comparison_test.csv",
        index=False,
    )

    predictions.to_csv(
        METRICS_DIR / "model_predictions.csv",
        index=False,
    )

    return test_results_df, predictions, fitted_models, best_name


# ============================================================
# 6. GRAFICO DEL CONFRONTO DEL MAE
# ============================================================

def plot_model_comparison(test_results_df):
    """Confronta il MAE dei modelli sul test set."""

    ordered = test_results_df.sort_values("MAE")

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        ordered["Model"],
        ordered["MAE"],
        edgecolor="black",
    )

    plt.bar_label(
        bars,
        fmt="%.2f",
        padding=3,
    )

    plt.ylabel("MAE sul test set")
    plt.xlabel("Modello")
    plt.title("Confronto dei modelli di regressione")
    plt.xticks(rotation=15)
    plt.tight_layout()

    output_path = FIGURES_DIR / "model_comparison_mae.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nGrafico salvato: {output_path}")


# ============================================================
# 7. GRAFICO DELLE PREVISIONI
# ============================================================

def plot_predictions(predictions, best_name):
    """Mostra le previsioni del modello selezionato dalla CV."""

    actual = predictions["Actual"]
    predicted = predictions[best_name]

    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())

    plt.figure(figsize=(8, 7))

    plt.scatter(
        actual,
        predicted,
        alpha=0.75,
        edgecolors="black",
    )

    plt.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        label="Previsione ideale",
    )

    plt.xlabel("Valore osservato")
    plt.ylabel("Valore previsto")
    plt.title(f"Previsioni sul test set: {best_name}")
    plt.legend()
    plt.tight_layout()

    output_path = FIGURES_DIR / "best_model_predictions.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Grafico salvato: {output_path}")


# ============================================================
# 8. PROGRAMMA PRINCIPALE
# ============================================================

def main():

    print("=" * 70)
    print("DIABETES REGRESSION AI - MODEL COMPARISON")
    print("=" * 70)

    X_train, X_test, y_train, y_test = load_and_split_data()

    print(f"\nTraining set: {len(X_train)} osservazioni")
    print(f"Test set:     {len(X_test)} osservazioni")

    models = create_models()

    cv_results = evaluate_cross_validation(
        models,
        X_train,
        y_train,
    )

    test_results, predictions, _, best_name = evaluate_test_set(
        models,
        cv_results,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\nRIEPILOGO CROSS-VALIDATION")
    print("=" * 70)
    print(
        cv_results.sort_values("CV_MAE_mean")
        .round(3)
        .to_string(index=False)
    )

    print("\nRIEPILOGO TEST SET")
    print("=" * 70)
    print(
        test_results.sort_values("MAE")
        .round(3)
        .to_string(index=False)
    )

    plot_model_comparison(test_results)
    plot_predictions(predictions, best_name)

    print("\nFILE GENERATI")
    print(f"Metriche: {METRICS_DIR}")
    print(f"Grafici:  {FIGURES_DIR}")

    print("\nCONFRONTO COMPLETATO")


if __name__ == "__main__":
    main()
