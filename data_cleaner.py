"""DataCleaner: deduplicate, treat outliers, derive analysis-ready frames."""
from __future__ import annotations

import numpy as np
import pandas as pd


class DataCleaner:
    """Clean a loaded HK property DataFrame and split it into Sale/Rental views.

    Outlier rule mirrors the AMS3640 reference: |x - mean| > 3*SD.
    For sale records the price column lives in HKD millions and the unit_rate
    is HKD per saleable square foot (lump-sum). For rental records the price
    column is HKD per month and the unit_rate is HKD per saleable sqft per month.
    The two regimes are separated before any numeric inference so that the
    different magnitudes never get pooled together.
    """

    def __init__(self, df: pd.DataFrame):
        self.df_raw = df.copy()
        self.report: dict = {}

    @staticmethod
    def _three_sigma_mask(series: pd.Series) -> pd.Series:
        mu, sd = series.mean(), series.std(ddof=0)
        return (series - mu).abs() > 3 * sd

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in df.columns if c != "row_id"]
        before = len(df)
        out = df.drop_duplicates(subset=cols).reset_index(drop=True)
        self.report["duplicates_removed"] = before - len(out)
        return out

    def treat_outliers(self, df: pd.DataFrame, kind: str) -> pd.DataFrame:
        """Drop unit_rate / saleable_area outliers using the 3-sigma rule.

        We keep the floor outliers because high-rise floors are realistic in HK
        (mirrors the reference report's decision to keep the 'Floor' outliers).
        """
        out = df.copy()
        flags = {}
        for col in ("unit_rate", "saleable_area"):
            mask = self._three_sigma_mask(out[col])
            flags[col] = int(mask.sum())
            out = out[~mask]
        # Drop zero / non-positive area or rate (defensive)
        bad = (out["unit_rate"] <= 0) | (out["saleable_area"] <= 0)
        flags["non_positive"] = int(bad.sum())
        out = out[~bad].reset_index(drop=True)
        self.report[f"outliers_{kind}"] = flags
        self.report[f"rows_after_clean_{kind}"] = len(out)
        return out

    def split_market(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        sale = df[~df["is_rental"]].copy()
        rental = df[df["is_rental"]].copy()
        return sale, rental

    def run(self) -> dict[str, pd.DataFrame]:
        df = self.remove_duplicates(self.df_raw)
        sale, rental = self.split_market(df)
        sale_clean = self.treat_outliers(sale, "sale")
        rental_clean = self.treat_outliers(rental, "rental")
        # Derive a price-index frame: monthly mean unit_rate per district + market.
        sale_clean["market"] = "Sale"
        rental_clean["market"] = "Rental"
        self.report["rows_raw"] = len(self.df_raw)
        self.report["rows_after_dedup"] = len(df)
        return {
            "all": df,
            "sale": sale_clean,
            "rental": rental_clean,
        }

    @staticmethod
    def to_price_index(sale: pd.DataFrame, rental: pd.DataFrame) -> pd.DataFrame:
        """Build a long-form monthly price index per district & market."""
        def agg(df: pd.DataFrame, label: str) -> pd.DataFrame:
            g = (
                df.groupby(["year_month", "district"])
                .agg(unit_rate_mean=("unit_rate", "mean"),
                     unit_rate_median=("unit_rate", "median"),
                     n=("unit_rate", "size"))
                .reset_index()
            )
            g["market"] = label
            return g
        return pd.concat([agg(sale, "Sale"), agg(rental, "Rental")], ignore_index=True)
