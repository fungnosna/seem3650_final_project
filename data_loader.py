"""DataLoader: read the raw CSV and return a typed DataFrame."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


class DataLoader:
    """Load the HK property pricing CSV and apply basic type coercion."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        df = df.rename(columns={
            "Unnamed: 0": "row_id",
            "saleable_area(ft^2)": "saleable_area_raw",
            "Public Housing": "public_housing",
            "Rental": "is_rental",
        })
        df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
        df["saleable_area"] = (
            df["saleable_area_raw"].astype(str).str.replace(",", "", regex=False).astype(float)
        )
        df["change_pct"] = (
            df["changes"].astype(str).str.replace("%", "", regex=False).replace("--", None)
        )
        df["change_pct"] = pd.to_numeric(df["change_pct"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["year_month"] = df["date"].dt.to_period("M").astype(str)
        self._df = df
        return df

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            return self.load()
        return self._df

    def summary(self) -> dict:
        df = self.df
        return {
            "rows": len(df),
            "cols": df.shape[1],
            "date_min": df["date"].min(),
            "date_max": df["date"].max(),
            "n_districts": df["district"].nunique(),
            "n_rental": int(df["is_rental"].sum()),
            "n_sale": int((~df["is_rental"]).sum()),
            "n_public_housing": int(df["public_housing"].sum()),
        }
