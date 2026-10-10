
"""
DIABETES REGRESSION AI
Fase 1: esplorazione del dataset.

Obiettivi:
- Caricare il dataset Diabetes di scikit-learn
- Analizzare dimensioni e caratteristiche
- Verificare valori mancanti e statistiche descrittive
- Visualizzare le distribuzioni delle variabili
- Esaminare la variabile target
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes


# ============================================================
# 1. CONFIGURAZIONE DEL PROGETTO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


# ============================================================
# 2. CARICAMENTO DEL DATASET
# ============================================================

def load_dataset():
    """Carica il dataset Diabetes incluso in scikit-learn."""

    dataset = load_diabetes(as_frame=True)

    X = dataset.data
    y = dataset.target

    df = X.copy()
    df["target"] = y

    return dataset, X, y, df


# ============================================================
# 3. ANALISI GENERALE
# ============================================================

def analyze_dataset(dataset, X, y, df):
    """Stampa le informazioni principali del dataset."""

    print("\n" + "=" * 65)
    print("DIABETES REGRESSION AI - DATA EXPLORATION")
    print("=" * 65)

    print("\n1. DIMENSIONI DEL DATASET")
    print("-" * 65)
    print(f"Numero di pazienti: {X.shape[0]}")
    print(f"Numero di caratteristiche: {X.shape[1]}")
    print(f"Numero totale di colonne con target: {df.shape[1]}")

    print("\n2. NOMI DELLE CARATTERISTICHE")
    print("-" * 65)

    for index, feature in enumerate(X.columns, start=1):
        print(f"{index:2}. {feature}")

    print("\n3. PRIME CINQUE OSSERVAZIONI")
    print("-" * 65)
    print(df.head().to_string())

    print("\n4. INFORMAZIONI SULLE VARIABILI")
    print("-" * 65)
    print(df.info())

    print("\n5. VALORI MANCANTI")
    print("-" * 65)
    print(df.isnull().sum())

    print("\n6. STATISTICHE DESCRITTIVE")
    print("-" * 65)
    print(df.describe().round(3).to_string())

    print("\n7. DESCRIZIONE DEL DATASET")
    print("-" * 65)
    print(dataset.DESCR)

    print("\n8. DESCRIZIONE DEL TARGET")
    print("-" * 65)
    print(f"Media: {y.mean():.2f}")
    print(f"Deviazione standard: {y.std():.2f}")
    print(f"Valore minimo: {y.min():.2f}")
    print(f"Valore massimo: {y.max():.2f}")


# ============================================================
# 4. VISUALIZZAZIONE DEL TARGET
# ============================================================

def plot_target_distribution(y):
    """Visualizza la distribuzione del valore target."""

    plt.figure(figsize=(9, 5))

    sns.histplot(y, bins=25, kde=True)

    plt.title("Distribuzione del target - Progressione del diabete")
    plt.xlabel("Valore target")
    plt.ylabel("Numero di osservazioni")
    plt.tight_layout()

    output_path = FIGURES_DIR / "target_distribution.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nGrafico salvato: {output_path}")

    plt.show()
    plt.close()


# ============================================================
# 5. VISUALIZZAZIONE DELLE CARATTERISTICHE
# ============================================================

def plot_feature_distributions(X):
    """Crea un grafico con la distribuzione di ogni caratteristica."""

    axes = X.hist(
        bins=25,
        figsize=(14, 10),
        layout=(3, 4),
        edgecolor="black"
    )

    plt.suptitle(
        "Distribuzione delle caratteristiche cliniche",
        fontsize=16
    )

    plt.tight_layout()

    output_path = FIGURES_DIR / "feature_distributions.png"
    plt.savefig(output_path, dpi=150)
    print(f"Grafici salvati: {output_path}")

    plt.show()
    plt.close()


# ============================================================
# 6. CORRELAZIONI
# ============================================================

def plot_correlation_matrix(df):
    """Visualizza le correlazioni tra caratteristiche e target."""

    correlation = df.corr(numeric_only=True)

    plt.figure(figsize=(11, 8))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True
    )

    plt.title("Matrice di correlazione")
    plt.tight_layout()

    output_path = FIGURES_DIR / "correlation_matrix.png"
    plt.savefig(output_path, dpi=150)
    print(f"Grafico salvato: {output_path}")

    plt.show()
    plt.close()


# ============================================================
# 7. PROGRAMMA PRINCIPALE
# ============================================================

def main():
    dataset, X, y, df = load_dataset()

    analyze_dataset(dataset, X, y, df)

    plot_target_distribution(y)

    plot_feature_distributions(X)

    plot_correlation_matrix(df)

    print("\n" + "=" * 65)
    print("ESPLORAZIONE COMPLETATA")
    print(f"Risultati salvati in: {RESULTS_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
