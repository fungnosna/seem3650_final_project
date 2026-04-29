"""Plotter: produce all PNG figures into result/graph."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import FIG_SIZE_DEFAULT, FIG_SIZE_TALL, FIG_SIZE_WIDE, PLOT_DPI


class Plotter:
    """Produce all figures used by the report."""

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")
        self.saved: list[str] = []

    # -- helpers
    def _save(self, name: str) -> str:
        path = self.out_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        self.saved.append(name)
        return str(path)

    # ---------------------------------------------------------- distributions
    def price_distributions(self, sale: pd.DataFrame, rental: pd.DataFrame) -> None:
        fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE_WIDE)
        sns.histplot(sale["unit_rate"], bins=60, ax=axes[0], color="steelblue")
        axes[0].set_title("Sale unit rate (HKD / sqft)")
        axes[0].set_xlabel("HKD per sqft")
        sns.histplot(rental["unit_rate"], bins=60, ax=axes[1], color="darkorange")
        axes[1].set_title("Rental unit rate (HKD / sqft / month)")
        axes[1].set_xlabel("HKD per sqft per month")
        self._save("01_unit_rate_distributions.png")

    def area_floor_distributions(self, sale: pd.DataFrame) -> None:
        fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE_WIDE)
        sns.histplot(sale["saleable_area"], bins=60, ax=axes[0], color="seagreen")
        axes[0].set_title("Saleable area distribution (sale)")
        axes[0].set_xlabel("ft²")
        sns.histplot(sale["floor"], bins=40, ax=axes[1], color="purple")
        axes[1].set_title("Floor distribution (sale)")
        axes[1].set_xlabel("Floor")
        self._save("02_area_floor_distributions.png")

    # ------------------------------------------------------ time-series index
    def price_index_over_time(self, sale: pd.DataFrame, rental: pd.DataFrame) -> None:
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for ax, df, label, color in (
            (axes[0], sale, "Sale (HKD / sqft)", "steelblue"),
            (axes[1], rental, "Rental (HKD / sqft / month)", "darkorange"),
        ):
            monthly = df.groupby("year_month")["unit_rate"].mean()
            ax.plot(monthly.index, monthly.values, marker="o", color=color, lw=1.4)
            ax.set_title(f"Mean unit rate over time — {label}")
            ax.set_ylabel("Mean unit rate")
            ax.tick_params(axis="x", rotation=60)
        self._save("03_price_index_over_time.png")

    def yearly_box(self, sale: pd.DataFrame) -> None:
        plt.figure(figsize=FIG_SIZE_DEFAULT)
        sns.boxplot(
            data=sale, x="year", y="unit_rate", hue="year",
            palette="Blues", showfliers=False, legend=False,
        )
        plt.title("Sale unit rate by year")
        plt.ylabel("HKD per sqft")
        self._save("04_unit_rate_by_year.png")

    # ------------------------------------------------------- district effects
    def district_box(self, sale: pd.DataFrame) -> None:
        order = (
            sale.groupby("district")["unit_rate"].median().sort_values(ascending=False).index
        )
        plt.figure(figsize=(11, 6))
        sns.boxplot(
            data=sale,
            y="district",
            x="unit_rate",
            order=order,
            hue="district",
            palette="viridis",
            showfliers=False,
            legend=False,
        )
        plt.title("Sale unit rate by district (median-ordered)")
        plt.xlabel("HKD per sqft")
        plt.ylabel("")
        self._save("05_unit_rate_by_district.png")

    def heatmap_district_year(self, sale: pd.DataFrame) -> None:
        piv = sale.pivot_table(
            index="district", columns="year", values="unit_rate", aggfunc="mean"
        )
        piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
        plt.figure(figsize=(8, 7))
        sns.heatmap(piv, annot=True, fmt=".0f", cmap="YlOrRd")
        plt.title("Mean sale unit rate (HKD/sqft) — district × year")
        plt.xlabel("Year")
        plt.ylabel("")
        self._save("06_heatmap_district_year.png")

    # ------------------------------------------------------------- regression
    def correlation_matrix(self, sale: pd.DataFrame) -> None:
        cols = ["unit_rate", "saleable_area", "floor", "year", "month"]
        corr = sale[cols].corr()
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
        plt.title("Correlation — numeric features (sale)")
        self._save("07_correlation_matrix.png")

    def actual_vs_predicted(self, y_true: pd.Series, y_pred: np.ndarray) -> None:
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.2, s=8, color="steelblue")
        lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        plt.plot(lim, lim, "r--", lw=1)
        plt.xlabel("Actual unit rate (HKD/sqft)")
        plt.ylabel("Predicted unit rate")
        plt.title("Backward-OLS: actual vs predicted (test set)")
        self._save("08_regression_actual_vs_pred.png")

    def residuals(self, residuals: np.ndarray) -> None:
        fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE_WIDE)
        sns.histplot(residuals, bins=60, kde=True, ax=axes[0], color="slategray")
        axes[0].set_title("Residual distribution")
        axes[0].set_xlabel("Residual (HKD/sqft)")
        from scipy import stats

        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title("Q-Q plot of residuals")
        self._save("09_regression_residuals.png")

    # ----------------------------------------------------------- classification
    def classification_metrics(self, summary: pd.DataFrame) -> None:
        plt.figure(figsize=FIG_SIZE_DEFAULT)
        m = summary.melt(
            id_vars="model",
            value_vars=["kappa"],
            var_name="metric",
            value_name="value",
        )
        sns.barplot(
            data=summary, x="model", y="kappa", hue="model",
            palette="crest", legend=False,
        )
        plt.title("Classifier comparison — Cohen's Kappa")
        plt.ylabel("Kappa")
        plt.xticks(rotation=15)
        self._save("10_classification_kappa.png")

    def confusion_matrix_rf(self, cm: np.ndarray, labels: list[str]) -> None:
        plt.figure(figsize=(5.5, 4.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Random Forest — confusion matrix")
        self._save("11_rf_confusion_matrix.png")

    def rf_importance(self, importances: pd.Series) -> None:
        plt.figure(figsize=(8, 5))
        importances.sort_values().plot.barh(color="cadetblue")
        plt.title("Random Forest — feature importance (top 15)")
        plt.xlabel("Importance")
        self._save("12_rf_feature_importance.png")

    # ----------------------------------------------------------------- cluster
    def elbow(self, inertias: dict[int, float], k_chosen: int) -> None:
        ks = sorted(inertias)
        plt.figure(figsize=FIG_SIZE_DEFAULT)
        plt.plot(ks, [inertias[k] for k in ks], marker="o", color="steelblue")
        plt.axvline(k_chosen, color="red", linestyle="--", label=f"k chosen = {k_chosen}")
        plt.title("K-means inertia (elbow plot)")
        plt.xlabel("k")
        plt.ylabel("Inertia (within-cluster SS)")
        plt.legend()
        self._save("13_kmeans_elbow.png")

    def cluster_scatter(self, df_labelled: pd.DataFrame) -> None:
        plt.figure(figsize=(8, 6))
        palette = sns.color_palette("tab10", df_labelled["cluster"].nunique())
        sns.scatterplot(
            data=df_labelled,
            x="saleable_area",
            y="unit_rate",
            hue="cluster",
            palette=palette,
            s=10,
            alpha=0.5,
            edgecolor=None,
        )
        plt.title("K-means clusters — saleable area vs unit rate")
        plt.xlabel("Saleable area (ft²)")
        plt.ylabel("Unit rate (HKD/sqft)")
        plt.legend(title="Cluster")
        self._save("14_kmeans_clusters.png")

    # ---------------------------------------------------- public housing flag
    def public_housing_compare(self, sale: pd.DataFrame) -> None:
        plt.figure(figsize=FIG_SIZE_DEFAULT)
        sns.boxplot(
            data=sale, x="public_housing", y="unit_rate", hue="public_housing",
            palette="pastel", showfliers=False, legend=False,
        )
        plt.xticks([0, 1], ["Private", "Public housing"])
        plt.title("Sale unit rate — private vs public housing")
        plt.ylabel("HKD per sqft")
        plt.xlabel("")
        self._save("15_public_vs_private.png")
