from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor

_DATA_DIR = Path(__file__).parent.parent / "data"

"""“On a remplacé une règle certaine… par une décision probabiliste observable.”"""
class ETAPE2_modelML_noRuleBased:

    def __init__(self, evollis_path: str, hybris_path: str):
        self.evollis_path = evollis_path
        self.hybris_path = hybris_path

        self.ORIGIN_LABELS = {
            0: "EVOLLIS",
            1: "HYBRIS"
        }

        # ➜ Features simples + signaux faibles
        self.FEATURES = [
            "quantity",
            "unit_price_EVOLLIS",
            "unit_price_HYBRIS",
            "total_price_EVOLLIS",
            "total_price_HYBRIS",
            "gap_evollis",
            "gap_hybris"
        ]

        self.clf_origin = None

    # ============================================================
    # 1. LOAD
    # ============================================================
    def load_data(self):
        self.df_evollis = pd.read_csv(self.evollis_path, sep=";")
        self.df_hybris = pd.read_csv(self.hybris_path, sep=";")

    # ============================================================
    # 2. MERGE + FEATURES
    # ============================================================
    def prepare_data(self):
        self.df = self.df_evollis.merge(
            self.df_hybris,
            on=["order_id", "quantity", "product_name", "order_date"],
            suffixes=("_EVOLLIS", "_HYBRIS")
        )

        self.df["expected_price_evollis"] = (
            self.df["quantity"] * self.df["unit_price_EVOLLIS"]
        )
        self.df["expected_price_hybris"] = (
            self.df["quantity"] * self.df["unit_price_HYBRIS"]
        )

        # ➜ Signaux faibles (aucune règle dure)
        self.df["gap_evollis"] = abs(
            self.df["total_price_EVOLLIS"] - self.df["expected_price_evollis"]
        )
        self.df["gap_hybris"] = abs(
            self.df["total_price_HYBRIS"] - self.df["expected_price_hybris"]
        )

        # Détection incohérence (on garde la règle)
        self.df["is_incoherent"] = (
            (self.df["gap_evollis"] > 0) | (self.df["gap_hybris"] > 0)
        ).astype(int)

    # ============================================================
    # 3. ENTRAÎNEMENT ORIGINE (LABEL FAIBLE)
    # ============================================================
    def train_origin_model(self):
        df_train = self.df[self.df["is_incoherent"] == 1].copy()

        # ➜ Label faible : qui est le PLUS suspect (mais bruité)
        df_train["weak_origin"] = np.where(
            df_train["gap_evollis"] >= df_train["gap_hybris"],
            0,  # EVOLLIS
            1   # HYBRIS
        )

        X = df_train[self.FEATURES]
        y = df_train["weak_origin"]

        self.clf_origin = XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=2,
            eval_metric="mlogloss",
            random_state=42
        )

        self.clf_origin.fit(X, y)

    # ============================================================
    # 4. PRÉDICTION AVEC PROBABILITÉS
    # ============================================================
    def predict(self):
        self.df_result = self.df.copy()

        mask = self.df_result["is_incoherent"] == 1
        X = self.df_result.loc[mask, self.FEATURES]

        probas = self.clf_origin.predict_proba(X)

        self.df_result.loc[mask, "p_origin_evollis"] = probas[:, 0]
        self.df_result.loc[mask, "p_origin_hybris"] = probas[:, 1]

        self.df_result["predicted_origin"] = np.where(
            self.df_result["p_origin_evollis"] >= self.df_result["p_origin_hybris"],
            "EVOLLIS",
            "HYBRIS"
        )

        self.df_result[["p_origin_evollis", "p_origin_hybris"]] = (
            self.df_result[["p_origin_evollis", "p_origin_hybris"]].fillna(0)
        )

    # ============================================================
    # 5. PIPELINE
    # ============================================================
    def run(self):
        self.load_data()
        self.prepare_data()
        self.train_origin_model()
        self.predict()
        return self.df_result

if __name__ == "__main__":
    workshop = ETAPE2_modelML_noRuleBased(
        evollis_path=str(_DATA_DIR / "evollis_products.csv"),
        hybris_path=str(_DATA_DIR / "hybris_products.csv")
    )

    df_final = workshop.run()
    df_final.to_csv(str(Path(__file__).parent / "ETAPE2_modeleML_noRuleBased.csv"), sep=";", index=False)

    print(df_final[[
        "order_id",
        "product_name",
        "is_incoherent",
        "p_origin_evollis",
        "p_origin_hybris",
        "predicted_origin"
    ]])