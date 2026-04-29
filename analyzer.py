"""StatAnalyzer: regression, statistical tests, classification, clustering."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from config import RANDOM_STATE


class StatAnalyzer:
    """Run all statistical analyses on the cleaned sale frame.

    Sale records are the analytical focus because their unit_rate is directly
    comparable to a price-per-sqft index. Rental records are summarised
    descriptively elsewhere.
    """

    PREDICTORS = [
        "saleable_area",
        "floor",
        "year",
        "month",
        "public_housing",
        "district",
    ]

    def __init__(self, sale: pd.DataFrame, rental: pd.DataFrame):
        self.sale = sale.reset_index(drop=True)
        self.rental = rental.reset_index(drop=True)
        self.results: dict = {}

    # ------------------------------------------------------------------ design
    def design_matrix(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        x = df[self.PREDICTORS].copy()
        x["public_housing"] = x["public_housing"].astype(int)
        x = pd.get_dummies(x, columns=["district"], drop_first=True, dtype=float)
        x = sm.add_constant(x.astype(float))
        y = df["unit_rate"].astype(float).reset_index(drop=True)
        return x, y

    # ------------------------------------------------------------- regression
    def regression(self) -> dict:
        x, y = self.design_matrix(self.sale)
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, test_size=0.25, random_state=RANDOM_STATE
        )
        full = sm.OLS(y_tr, x_tr).fit()
        # Backward elimination by highest p-value, stop at alpha = 0.05
        backward = self._backward_elimination(x_tr, y_tr, alpha=0.05)
        x_te_b = x_te[backward.model.exog_names]
        rmse_full = float(np.sqrt(np.mean((y_te - full.predict(x_te)) ** 2)))
        rmse_back = float(np.sqrt(np.mean((y_te - backward.predict(x_te_b)) ** 2)))

        out = {
            "n_train": len(x_tr),
            "n_test": len(x_te),
            "full": {
                "n_estimators": int(full.df_model + 1),
                "r2": float(full.rsquared),
                "adj_r2": float(full.rsquared_adj),
                "aic": float(full.aic),
                "f": float(full.fvalue),
                "rmse_test": rmse_full,
            },
            "backward": {
                "n_estimators": int(backward.df_model + 1),
                "r2": float(backward.rsquared),
                "adj_r2": float(backward.rsquared_adj),
                "aic": float(backward.aic),
                "f": float(backward.fvalue),
                "rmse_test": rmse_back,
                "kept": list(backward.model.exog_names),
            },
            "coef_table": backward.summary2().tables[1].reset_index().rename(
                columns={"index": "variable"}
            ),
        }
        self.results["regression"] = out
        return out

    @staticmethod
    def _backward_elimination(x: pd.DataFrame, y: pd.Series, alpha: float = 0.05):
        cols = list(x.columns)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            while True:
                model = sm.OLS(y, x[cols]).fit()
                pvals = model.pvalues.drop(labels=["const"], errors="ignore")
                if pvals.empty or pvals.max() < alpha:
                    return model
                drop = pvals.idxmax()
                cols.remove(drop)
                if not cols:
                    return model

    # ------------------------------------------------------- statistical tests
    def two_sample_test(self) -> dict:
        med = self.sale["unit_rate"].median()
        top = self.sale.loc[self.sale["unit_rate"] >= med, "unit_rate"]
        bot = self.sale.loc[self.sale["unit_rate"] < med, "unit_rate"]
        t = stats.ttest_ind(top, bot, equal_var=False)
        ci95 = stats.t.interval(
            0.95,
            df=len(top) + len(bot) - 2,
            loc=top.mean() - bot.mean(),
            scale=np.sqrt(top.var(ddof=1) / len(top) + bot.var(ddof=1) / len(bot)),
        )
        out = {
            "t": float(t.statistic),
            "p": float(t.pvalue),
            "df": int(len(top) + len(bot) - 2),
            "mean_top": float(top.mean()),
            "mean_bot": float(bot.mean()),
            "sd_top": float(top.std(ddof=1)),
            "sd_bot": float(bot.std(ddof=1)),
            "ci95": (float(ci95[0]), float(ci95[1])),
        }
        self.results["t_test"] = out
        return out

    def anova_district(self) -> dict:
        groups = [g["unit_rate"].values for _, g in self.sale.groupby("district")]
        f, p = stats.f_oneway(*groups)
        means = (
            self.sale.groupby("district")["unit_rate"].mean().sort_values(ascending=False)
        )
        out = {
            "f": float(f),
            "p": float(p),
            "df_between": self.sale["district"].nunique() - 1,
            "df_within": len(self.sale) - self.sale["district"].nunique(),
            "ranking": means,
        }
        self.results["anova_district"] = out
        return out

    def anova_year(self) -> dict:
        groups = [g["unit_rate"].values for _, g in self.sale.groupby("year")]
        f, p = stats.f_oneway(*groups)
        out = {
            "f": float(f),
            "p": float(p),
            "yearly_mean": self.sale.groupby("year")["unit_rate"].mean(),
            "yearly_median": self.sale.groupby("year")["unit_rate"].median(),
        }
        self.results["anova_year"] = out
        return out

    # --------------------------------------------------------- classification
    def classify_price_tier(self, sample: int | None = 30000) -> dict:
        df = self.sale.copy()
        if sample is not None and len(df) > sample:
            df = df.sample(sample, random_state=RANDOM_STATE).reset_index(drop=True)
        q30, q70 = df["unit_rate"].quantile([0.30, 0.70])
        def tier(v):
            if v < q30:
                return "Low"
            if v < q70:
                return "Median"
            return "High"
        df["tier"] = df["unit_rate"].apply(tier)
        x = df[["saleable_area", "floor", "year", "month", "public_housing"]].copy()
        x["public_housing"] = x["public_housing"].astype(int)
        x = pd.concat(
            [x, pd.get_dummies(df["district"], prefix="dist", drop_first=True, dtype=float)],
            axis=1,
        )
        y = df["tier"]
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
        )
        scaler = StandardScaler().fit(x_tr)
        x_tr_s = scaler.transform(x_tr)
        x_te_s = scaler.transform(x_te)

        models = {
            "SVM (linear)": LinearSVC(C=1.0, random_state=RANDOM_STATE, max_iter=5000),
            "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
            "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "Random Forest": RandomForestClassifier(
                n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE
            ),
        }
        labels = ["High", "Median", "Low"]
        rows = []
        per_model = {}
        for name, model in models.items():
            tr_x = x_tr_s if name in ("SVM (linear)", "KNN (k=5)") else x_tr
            te_x = x_te_s if name in ("SVM (linear)", "KNN (k=5)") else x_te
            model.fit(tr_x, y_tr)
            pred = model.predict(te_x)
            kappa = cohen_kappa_score(y_te, pred)
            errors = int((pred != y_te).sum())
            rep = classification_report(
                y_te, pred, labels=labels, output_dict=True, zero_division=0
            )
            cm = confusion_matrix(y_te, pred, labels=labels)
            row = {
                "model": name,
                "kappa": kappa,
                "errors": errors,
                **{f"precision_{lbl}": rep[lbl]["precision"] for lbl in labels},
                **{f"recall_{lbl}": rep[lbl]["recall"] for lbl in labels},
            }
            rows.append(row)
            per_model[name] = {"cm": cm, "report": rep}

        rf = models["Random Forest"]
        importances = (
            pd.Series(rf.feature_importances_, index=x.columns)
            .sort_values(ascending=False)
            .head(15)
        )
        out = {
            "summary": pd.DataFrame(rows),
            "per_model": per_model,
            "rf_importance": importances,
            "labels": labels,
            "thresholds": {"q30": float(q30), "q70": float(q70)},
            "n_train": len(x_tr),
            "n_test": len(x_te),
        }
        self.results["classification"] = out
        return out

    # ----------------------------------------------------- association rules
    def association_rules(self) -> dict:
        """Lightweight Apriori-style scan over discretised attributes.

        We bin unit_rate, saleable_area, floor, year (5-year band) and merge with
        district and a high/low public_housing flag, then enumerate rules
        with support >= 0.05 and confidence >= 0.9 over itemsets up to size 3.
        """
        df = self.sale.copy()
        q30, q70 = df["unit_rate"].quantile([0.30, 0.70])
        a30, a70 = df["saleable_area"].quantile([0.30, 0.70])
        f30, f70 = df["floor"].quantile([0.30, 0.70])

        def tier(v, lo, hi):
            return "Low" if v < lo else ("Median" if v < hi else "High")

        df["price_tier"] = df["unit_rate"].apply(lambda v: tier(v, q30, q70))
        df["area_tier"] = df["saleable_area"].apply(lambda v: tier(v, a30, a70))
        df["floor_tier"] = df["floor"].apply(lambda v: tier(v, f30, f70))
        df["year_band"] = df["year"].astype(str)
        df["district_short"] = df["district"].str.replace(" District", "", regex=False)

        items_cols = ["price_tier", "area_tier", "floor_tier", "year_band", "district_short"]
        prefixed = pd.DataFrame(
            {c: c + "=" + df[c].astype(str) for c in items_cols},
            index=df.index,
        )
        baskets = prefixed.values.tolist()
        n = len(baskets)

        # Frequency tables
        from collections import Counter
        from itertools import combinations

        c1 = Counter(it for basket in baskets for it in basket)
        L1 = {(it,): cnt / n for it, cnt in c1.items() if cnt / n >= 0.05}

        def count_pairs(size: int):
            counter = Counter()
            for basket in baskets:
                for combo in combinations(sorted(basket), size):
                    counter[combo] += 1
            return {k: v / n for k, v in counter.items() if v / n >= 0.05}

        L2 = count_pairs(2)
        L3 = count_pairs(3)

        rules = []
        for itemset, sup in {**L2, **L3}.items():
            for k in range(1, len(itemset)):
                for lhs in combinations(itemset, k):
                    rhs = tuple(sorted(set(itemset) - set(lhs)))
                    sup_lhs = L1.get(lhs) if k == 1 else L2.get(tuple(sorted(lhs)))
                    if sup_lhs is None:
                        continue
                    conf = sup / sup_lhs
                    if conf < 0.9:
                        continue
                    sup_rhs = L1.get(rhs) if len(rhs) == 1 else L2.get(rhs)
                    if not sup_rhs:
                        continue
                    lift = conf / sup_rhs
                    rules.append({
                        "lhs": ", ".join(lhs),
                        "rhs": ", ".join(rhs),
                        "support": sup,
                        "confidence": conf,
                        "lift": lift,
                    })

        rules_df = pd.DataFrame(
            rules, columns=["lhs", "rhs", "support", "confidence", "lift"]
        )
        if not rules_df.empty:
            rules_df = (
                rules_df.sort_values(
                    ["confidence", "lift", "support"], ascending=False
                ).reset_index(drop=True)
            )
        # If the strict thresholds returned nothing, retry with a relaxed
        # confidence floor so we always have something to discuss.
        if rules_df.empty:
            relaxed = []
            for itemset, sup in {**L2, **L3}.items():
                for k in range(1, len(itemset)):
                    for lhs in combinations(itemset, k):
                        rhs = tuple(sorted(set(itemset) - set(lhs)))
                        sup_lhs = L1.get(lhs) if k == 1 else L2.get(tuple(sorted(lhs)))
                        if not sup_lhs:
                            continue
                        conf = sup / sup_lhs
                        sup_rhs = L1.get(rhs) if len(rhs) == 1 else L2.get(rhs)
                        if not sup_rhs:
                            continue
                        relaxed.append({
                            "lhs": ", ".join(lhs),
                            "rhs": ", ".join(rhs),
                            "support": sup,
                            "confidence": conf,
                            "lift": conf / sup_rhs,
                        })
            rules_df = pd.DataFrame(
                relaxed, columns=["lhs", "rhs", "support", "confidence", "lift"]
            )
            if not rules_df.empty:
                rules_df = rules_df.sort_values(
                    ["confidence", "lift", "support"], ascending=False
                ).reset_index(drop=True)

        def _filter_rhs(target: str) -> pd.DataFrame:
            if rules_df.empty:
                return rules_df
            return rules_df[rules_df["rhs"] == target].head(10)

        relaxed_used = bool(
            rules_df.shape[0]
            and (rules_df["confidence"].min() < 0.9)
        )
        out = {
            "n_rules": len(rules_df),
            "rules": rules_df,
            "high_price_drivers": _filter_rhs("price_tier=High"),
            "low_price_drivers": _filter_rhs("price_tier=Low"),
            "settings": {
                "min_support": 0.05,
                "min_confidence_target": 0.9,
                "relaxed_to_min_support_only": relaxed_used,
            },
        }
        self.results["association"] = out
        return out

    # --------------------------------------------------------------- clustering
    def cluster(self, k_range=range(2, 8), sample: int | None = 20000) -> dict:
        df = self.sale[["unit_rate", "saleable_area", "floor"]].copy()
        if sample is not None and len(df) > sample:
            df = df.sample(sample, random_state=RANDOM_STATE).reset_index(drop=True)
        x = StandardScaler().fit_transform(df.values)

        inertias = {}
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(x)
            inertias[k] = float(km.inertia_)

        # Pick k via "elbow" — largest second derivative drop
        ks = sorted(inertias)
        diffs = np.diff([inertias[k] for k in ks])
        diffs2 = np.diff(diffs)
        k_best = ks[int(np.argmax(diffs2)) + 1]

        km = KMeans(n_clusters=k_best, n_init=20, random_state=RANDOM_STATE).fit(x)
        df["cluster"] = km.labels_
        profile = (
            df.groupby("cluster")
            .agg(
                n=("unit_rate", "size"),
                unit_rate=("unit_rate", "mean"),
                saleable_area=("saleable_area", "mean"),
                floor=("floor", "mean"),
            )
            .round(2)
        )
        out = {
            "k_range": list(k_range),
            "inertias": inertias,
            "k_chosen": k_best,
            "profile": profile,
            "labelled": df,
        }
        self.results["cluster"] = out
        return out
