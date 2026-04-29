"""Pipeline entry point — run end-to-end analysis."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import statsmodels.api as sm

# Ensure local modules import even when this is run from elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyzer import StatAnalyzer
from config import DATA_PATH, GRAPH_DIR, REPORT_DIR
from data_cleaner import DataCleaner
from data_loader import DataLoader
from plotter import Plotter
from reporter import Reporter


class Pipeline:
    """End-to-end orchestration: load -> clean -> analyse -> plot -> report."""

    def __init__(self):
        self.loader = DataLoader(DATA_PATH)
        self.plotter = Plotter(GRAPH_DIR)
        self.reporter = Reporter(REPORT_DIR, GRAPH_DIR)

    def run(self) -> dict:
        print("[1/6] loading raw data ...")
        self.loader.load()
        loader_summary = self.loader.summary()

        print("[2/6] cleaning ...")
        cleaner = DataCleaner(self.loader.df)
        frames = cleaner.run()
        sale, rental = frames["sale"], frames["rental"]
        clean_report = cleaner.report

        print("[3/6] plotting descriptives ...")
        self.plotter.price_distributions(sale, rental)
        self.plotter.area_floor_distributions(sale)
        self.plotter.price_index_over_time(sale, rental)
        self.plotter.yearly_box(sale)
        self.plotter.district_box(sale)
        self.plotter.heatmap_district_year(sale)
        self.plotter.correlation_matrix(sale)
        self.plotter.public_housing_compare(sale)

        print("[4/6] regression ...")
        analyzer = StatAnalyzer(sale, rental)
        regression = analyzer.regression()
        # Plot residuals + actual vs predicted using the backward model on test set.
        x_full, y_full = analyzer.design_matrix(sale)
        from sklearn.model_selection import train_test_split
        from config import RANDOM_STATE

        x_tr, x_te, y_tr, y_te = train_test_split(
            x_full, y_full, test_size=0.25, random_state=RANDOM_STATE
        )
        kept = regression["backward"]["kept"]
        backward_model = sm.OLS(y_tr, x_tr[kept]).fit()
        y_pred = backward_model.predict(x_te[kept])
        residuals = (y_te - y_pred).values
        self.plotter.actual_vs_predicted(y_te, y_pred.values)
        self.plotter.residuals(residuals)

        print("[5/6] tests + classification + association + clustering ...")
        ttest = analyzer.two_sample_test()
        anova_district = analyzer.anova_district()
        anova_year = analyzer.anova_year()
        classification = analyzer.classify_price_tier()
        cm_rf = classification["per_model"]["Random Forest"]["cm"]
        self.plotter.classification_metrics(classification["summary"])
        self.plotter.confusion_matrix_rf(cm_rf, classification["labels"])
        self.plotter.rf_importance(classification["rf_importance"])

        association = analyzer.association_rules()
        cluster = analyzer.cluster()
        self.plotter.elbow(cluster["inertias"], cluster["k_chosen"])
        self.plotter.cluster_scatter(cluster["labelled"])

        print("[6/6] writing report ...")
        md_path = self.reporter.render_markdown(
            loader_summary,
            clean_report,
            regression,
            ttest,
            anova_district,
            anova_year,
            classification,
            association,
            cluster,
        )
        docx_path = self.reporter.render_docx(md_path)
        print(f"  -> markdown: {md_path}")
        print(f"  -> docx:     {docx_path}")
        print(f"  -> graphs:   {GRAPH_DIR}")
        return {
            "markdown": md_path,
            "docx": docx_path,
            "graphs_saved": self.plotter.saved,
        }


if __name__ == "__main__":
    Pipeline().run()
