
"""
DIABETES REGRESSION AI
Fase 4: ottimizzazione degli iperparametri.

Modelli ottimizzati:
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Linear Regression viene mantenuta come baseline.

Metodo:
- Training/test split riproducibile
- GridSearchCV con 5-fold cross-validation
- Selezione basata sul MAE medio
- Valutazione finale sul test set
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.datasets import load_diabetes
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
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
MODELS_DIR = PROJECT_ROOT / "models"

for directory in (METRICS_DIR, FIGURES_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5


# ============================================================
# 2. CARICAMENTO DEI DATI
# ============================================================

def load_and_split_data():
    """Carica il dataset e crea training e test set."""

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
# 3. DEFINIZIONE DEI MODELLI E DELLE GRIGLIE
# ============================================================

def create_searches():
    """Crea le ricerche degli iperparametri."""

    cv = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    searches = {
        "Ridge": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge()),
            ]),
            "params": {
                "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
        },

        "RandomForest": {
            "estimator": RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": [200, 300],
                "max_depth": [None, 5, 10],
                "min_samples_leaf": [2, 4, 8],
            },
        },

        "GradientBoosting": {
            "estimator": GradientBoostingRegressor(
                random_state=RANDOM_STATE,
                loss="squared_error",
            ),
            "params": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [1, 2, 3],
                "min_samples_leaf": [3, 5],
            },
        },
    }

    return searches, cv


# ============================================================
# 4. OTTIMIZZAZIONE CON GRIDSEARCHCV
# ============================================================

def run_grid_searches(searches, cv, X_train, y_train):
    """Ottimizza i modelli usando esclusivamente il training set."""

    fitted_searches = {}
    cv_summaries = []
    detailed_results = []

    total_configurations = sum(
        len(search["params"]["model__alpha"])
        if name == "Ridge"
        else (
            len(search["params"]["n_estimators"])
            * len(search["params"]["max_depth"])
            * len(search["params"]["min_samples_leaf"])
            if name == "RandomForest"
            else (
                len(search["params"]["n_estimators"])
                * len(search["params"]["learning_rate"])
                * len(search["params"]["max_depth"])
                * len(search["params"]["min_samples_leaf"])
            )
        )
        for name, search in searches.items()
    )

    print(f"\nConfigurazioni totali da valutare: {total_configurations}")
    print(f"Fold di cross-validation: {N_SPLITS}")

    for name, search in searches.items():

        print("\n" + "=" * 70)
        print(f"GRID SEARCH: {name}")
        print("=" * 70)

        grid = GridSearchCV(
            estimator=search["estimator"],
            param_grid=search["params"],
            scoring="neg_mean_absolute_error",
            cv=cv,
            refit=True,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )

        grid.fit(X_train, y_train)

        fitted_searches[name] = grid

        best_mae = -grid.best_score_

        print(f"Miglior MAE medio CV: {best_mae:.3f}")
        print(f"Migliori parametri: {grid.best_params_}")

        cv_summaries.append({
            "Model": name,
            "Best_CV_MAE": best_mae,
            "CV_MAE_std": grid.cv_results_[
                "std_test_score"
            ][grid.best_index_],
            "Best_params": json.dumps(
                grid.best_params_,
                sort_keys=True,
            ),
        })

        # Conserva i risultati di tutte le configurazioni
        for index, params in enumerate(grid.cv_results_["params"]):
            detailed_results.append({
                "Model": name,
                "Mean_CV_MAE": -grid.cv_results_[
                    "mean_test_score"
                ][index],
                "Std_CV_MAE": grid.cv_results_[
                    "std_test_score"
                ][index],
                "Params": json.dumps(params, sort_keys=True),
            })

    cv_summary_df = pd.DataFrame(cv_summaries)
    cv_summary_df = cv_summary_df.sort_values("Best_CV_MAE")

    cv_summary_df.to_csv(
        METRICS_DIR / "optimized_cv_summary.csv",
        index=False,
    )

    pd.DataFrame(detailed_results).to_csv(
        METRICS_DIR / "grid_search_all_results.csv",
        index=False,
    )

    return fitted_searches, cv_summary_df


# ============================================================
# 5. VALUTAZIONE FINALE SUL TEST SET
# ============================================================

def evaluate_test_set(
    fitted_searches,
    cv_summary_df,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Confronta i modelli ottimizzati sul test set.
    La selezione del vincitore viene effettuata tramite CV.
    """

    # Linear Regression rimane il riferimento non ottimizzato
    models = {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])
    }

    for name, grid in fitted_searches.items():
        models[name] = grid.best_estimator_

    selected_name = cv_summary_df.iloc[0]["Model"]

    print("\n" + "=" * 70)
    print("MODELLO SELEZIONATO DALLA CROSS-VALIDATION")
    print("=" * 70)
    print(f"Modello: {selected_name}")
    print(
        f"MAE medio CV: "
        f"{cv_summary_df.iloc[0]['Best_CV_MAE']:.3f}"
    )

    test_rows = []
    predictions_df = pd.DataFrame({
        "Actual": y_test.to_numpy()
    })

    for name, model in models.items():

        # Clona il modello prima del fit finale
        final_model = clone(model)
        final_model.fit(X_train, y_train)

        y_pred = final_model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        test_rows.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Selected_by_CV": name == selected_name,
        })

        predictions_df[name] = y_pred

        print(f"\n{name}")
        print("-" * 50)
        print(f"MAE:  {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R²:   {r2:.3f}")

        # Salva il modello selezionato dalla CV
        if name == selected_name:
            import joblib

            model_path = MODELS_DIR / "best_regression_model.joblib"
            joblib.dump(final_model, model_path)
            print(f"Modello salvato: {model_path}")

    test_df = pd.DataFrame(test_rows)

    test_df = test_df.merge(
        cv_summary_df[["Model", "Best_CV_MAE"]],
        on="Model",
        how="left",
    )

    test_df.to_csv(
        METRICS_DIR / "optimized_test_metrics.csv",
        index=False,
    )

    predictions_df.to_csv(
        METRICS_DIR / "optimized_predictions.csv",
        index=False,
    )

    return test_df, predictions_df, selected_name


# ============================================================
# 6. GRAFICO DEL CONFRONTO
# ============================================================

def plot_comparison(test_df):
    """Crea un grafico del MAE sul test set."""

    ordered = test_df.sort_values("MAE")

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        ordered["Model"],
        ordered["MAE"],
        edgecolor="black",
    )

    plt.bar_label(bars, fmt="%.2f", padding=3)

    plt.ylabel("MAE sul test set")
    plt.xlabel("Modello")
    plt.title("Confronto: modelli ottimizzati")
    plt.xticks(rotation=15)
    plt.tight_layout()

    output_path = FIGURES_DIR / "optimized_model_comparison.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nGrafico salvato: {output_path}")


# ============================================================
# 7. GRAFICO DELLE PREVISIONI
# ============================================================

def plot_predictions(predictions_df, selected_name):
    """Visualizza le previsioni del modello selezionato via CV."""

    actual = predictions_df["Actual"]
    predicted = predictions_df[selected_name]

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
    plt.title(f"Modello selezionato: {selected_name}")
    plt.legend()
    plt.tight_layout()

    output_path = FIGURES_DIR / "optimized_best_predictions.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Grafico salvato: {output_path}")


# ============================================================
# 8. PROGRAMMA PRINCIPALE
# ============================================================

def main():

    print("=" * 70)
    print("DIABETES REGRESSION AI - HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)

    X_train, X_test, y_train, y_test = load_and_split_data()

    print(f"\nTraining set: {len(X_train)} osservazioni")
    print(f"Test set:     {len(X_test)} osservazioni")

    searches, cv = create_searches()

    fitted_searches, cv_summary_df = run_grid_searches(
        searches,
        cv,
        X_train,
        y_train,
    )

    test_df, predictions_df, selected_name = evaluate_test_set(
        fitted_searches,
        cv_summary_df,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\n" + "=" * 70)
    print("RIEPILOGO CROSS-VALIDATION")
    print("=" * 70)
    print(cv_summary_df.round(3).to_string(index=False))

    print("\n" + "=" * 70)
    print("RIEPILOGO TEST SET")
    print("=" * 70)
    print(
        test_df.sort_values("MAE")
        .round(3)
        .to_string(index=False)
    )

    plot_comparison(test_df)
    plot_predictions(predictions_df, selected_name)

    print("\nFILE GENERATI")
    print(f"Metriche: {METRICS_DIR}")
    print(f"Grafici:  {FIGURES_DIR}")
    print(f"Modello:  {MODELS_DIR}")

    print("\nOTTIMIZZAZIONE COMPLETATA")


if __name__ == "__main__":
    main()
