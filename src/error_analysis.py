
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# DIABETES REGRESSION AI - ERROR ANALYSIS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


def calculate_metrics(y_true, y_pred):
    """Calcola le principali metriche di regressione."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "Mean_Error": np.mean(y_pred - np.asarray(y_true)),
        "Median_Absolute_Error": np.median(
            np.abs(np.asarray(y_true) - np.asarray(y_pred))
        ),
        "Max_Absolute_Error": np.max(
            np.abs(np.asarray(y_true) - np.asarray(y_pred))
        ),
    }


def main():
    print("=" * 70)
    print("DIABETES REGRESSION AI - ERROR ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Caricamento dati e split riproducibile
    # --------------------------------------------------------
    dataset = load_diabetes(as_frame=True)
    X = dataset.data
    y = dataset.target

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
    # 2. Ricostruzione dei modelli già confrontati
    # --------------------------------------------------------
    models = {
        "LinearRegression": LinearRegression(),

        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),

        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=8,
            random_state=RANDOM_STATE,
            n_jobs=1
        ),

        "GradientBoosting": GradientBoostingRegressor(
            learning_rate=0.1,
            max_depth=1,
            min_samples_leaf=3,
            n_estimators=300,
            random_state=RANDOM_STATE
        ),
    }

    # --------------------------------------------------------
    # 3. Addestramento e previsioni
    # --------------------------------------------------------
    all_predictions = pd.DataFrame({
        "Actual": y_test.to_numpy()
    })

    metrics_rows = []

    for name, model in models.items():
        print(f"\nAnalisi modello: {name}")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        errors = predictions - y_test.to_numpy()

        all_predictions[f"{name}_Predicted"] = predictions
        all_predictions[f"{name}_Error"] = errors
        all_predictions[f"{name}_Absolute_Error"] = np.abs(errors)

        row = {"Model": name}
        row.update(calculate_metrics(y_test, predictions))
        metrics_rows.append(row)

        print(f"MAE:                 {row['MAE']:.3f}")
        print(f"RMSE:                {row['RMSE']:.3f}")
        print(f"R²:                  {row['R2']:.3f}")
        print(f"Errore medio:        {row['Mean_Error']:.3f}")
        print(f"Errore assoluto max: {row['Max_Absolute_Error']:.3f}")

    # --------------------------------------------------------
    # 4. Salvataggio metriche e previsioni
    # --------------------------------------------------------
    metrics_df = pd.DataFrame(metrics_rows).sort_values("MAE")

    metrics_path = METRICS_DIR / "error_analysis_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    predictions_path = METRICS_DIR / "error_analysis_predictions.csv"
    all_predictions.to_csv(predictions_path, index=False)

    # Top 10 errori del Gradient Boosting
    gb_errors = all_predictions[
        ["Actual", "GradientBoosting_Predicted",
         "GradientBoosting_Error", "GradientBoosting_Absolute_Error"]
    ].copy()

    gb_errors = gb_errors.sort_values(
        "GradientBoosting_Absolute_Error",
        ascending=False
    ).head(10)

    top_errors_path = METRICS_DIR / "gradient_boosting_top10_errors.csv"
    gb_errors.to_csv(top_errors_path, index=False)

    print("\n" + "=" * 70)
    print("CONFRONTO METRICHE")
    print("=" * 70)
    print(metrics_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}"
    ))

    print("\nTOP 10 ERRORI - GRADIENT BOOSTING")
    print("-" * 70)
    print(gb_errors.to_string(
        index=False,
        float_format=lambda value: f"{value:.2f}"
    ))

    # --------------------------------------------------------
    # 5. Grafico: valori reali vs predetti
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))

    for name in models:
        ax.scatter(
            all_predictions["Actual"],
            all_predictions[f"{name}_Predicted"],
            alpha=0.65,
            s=28,
            label=name
        )

    minimum = min(
        all_predictions["Actual"].min(),
        all_predictions[
            [f"{name}_Predicted" for name in models]
        ].min().min()
    )

    maximum = max(
        all_predictions["Actual"].max(),
        all_predictions[
            [f"{name}_Predicted" for name in models]
        ].max().max()
    )

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        color="black",
        label="Previsione ideale"
    )

    ax.set_title("Valori reali vs predetti - confronto modelli")
    ax.set_xlabel("Valore reale")
    ax.set_ylabel("Valore predetto")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    actual_predicted_path = FIGURES_DIR / "actual_vs_predicted_all_models.png"
    fig.savefig(actual_predicted_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------
    # 6. Grafico dei residui per Gradient Boosting
    # Residuo = valore reale - previsione
    # --------------------------------------------------------
    gb_predicted = all_predictions["GradientBoosting_Predicted"]
    residuals = all_predictions["Actual"] - gb_predicted

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        gb_predicted,
        residuals,
        alpha=0.7,
        s=32
    )

    ax.axhline(0, linestyle="--", color="black")
    ax.set_title("Analisi dei residui - Gradient Boosting")
    ax.set_xlabel("Valore predetto")
    ax.set_ylabel("Residuo (reale - predetto)")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    residuals_path = FIGURES_DIR / "gradient_boosting_residuals.png"
    fig.savefig(residuals_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------
    # 7. Distribuzione degli errori
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))

    for name in models:
        errors = all_predictions[f"{name}_Error"]
        ax.hist(
            errors,
            bins=18,
            alpha=0.35,
            label=name
        )

    ax.axvline(0, linestyle="--", color="black")
    ax.set_title("Distribuzione degli errori di previsione")
    ax.set_xlabel("Errore (predetto - reale)")
    ax.set_ylabel("Numero di osservazioni")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()

    error_distribution_path = FIGURES_DIR / "error_distribution_all_models.png"
    fig.savefig(error_distribution_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------
    # 8. Salvataggio riepilogo testuale
    # --------------------------------------------------------
    best_by_mae = metrics_df.iloc[0]

    report_path = METRICS_DIR / "error_analysis_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("DIABETES REGRESSION AI - ERROR ANALYSIS\n")
        file.write("=" * 60 + "\n\n")
        file.write(f"Osservazioni nel test set: {len(y_test)}\n\n")
        file.write("Metriche dei modelli:\n")
        file.write(metrics_df.to_string(index=False))
        file.write("\n\n")
        file.write(
            f"Miglior MAE sul test set: {best_by_mae['Model']}\n"
        )
        file.write(f"MAE: {best_by_mae['MAE']:.4f}\n")
        file.write(
            "\nNota: il test set è stato consultato nelle fasi precedenti. "
            "I risultati sono utili per l'analisi didattica, ma non "
            "costituiscono una valutazione finale completamente indipendente.\n"
        )

    print("\n" + "=" * 70)
    print("FILE GENERATI")
    print("=" * 70)
    print(f"Metriche:              {metrics_path}")
    print(f"Previsioni:            {predictions_path}")
    print(f"Top 10 errori:         {top_errors_path}")
    print(f"Report:                {report_path}")
    print(f"Reale vs predetto:     {actual_predicted_path}")
    print(f"Grafico residui:       {residuals_path}")
    print(f"Distribuzione errori:  {error_distribution_path}")

    print("\nANALISI DEGLI ERRORI COMPLETATA")


if __name__ == "__main__":
    main()
