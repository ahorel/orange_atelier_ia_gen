from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import os
import joblib

_DATA_DIR   = Path(__file__).parent.parent / "data"
_MODELS_DIR = Path(__file__).parent / "models"


class ETAPE5_modelML_warmStart:
    """
    Entraînement incrémental XGBoost (vrai warm start)
    Le modèle s'améliore en probabilité en voyant plus de cas réels
    """

    # ============================================================
    # INIT
    # ============================================================
    def __init__(self, evollis_path: str, hybris_path: str):
        self.evollis_path = evollis_path
        self.hybris_path = hybris_path

        self.ORIGIN_LABELS = {0: "EVOLLIS", 1: "HYBRIS"}

        self.FEATURES = [
            "quantity",
            "unit_price_EVOLLIS",
            "unit_price_HYBRIS",
            "total_price_EVOLLIS",
            "total_price_HYBRIS",
            "gap_evollis",
            "gap_hybris"
        ]

        self.model_path = str(_MODELS_DIR / "xgb_origin_model.pkl")
        self.n_estimators_step = 50
        self.clf_origin = None
        self.xgb_model = None

        self._load_or_init_model()

    # ============================================================
    # MODEL LOAD / INIT (🔥 CLÉ DU VRAI WARM START)
    # ============================================================
    def _load_or_init_model(self):
        if os.path.exists(self.model_path):
            print("🔁 Chargement du modèle existant (warm start réel)")
            self.clf_origin = joblib.load(self.model_path)

            # ➜ on ajoute des arbres
            self.clf_origin.n_estimators += self.n_estimators_step

            # ➜ booster existant
            self.xgb_model = self.clf_origin.get_booster()

        else:
            print("🆕 Création du modèle initial")
            self.clf_origin = XGBClassifier(
                n_estimators=self.n_estimators_step,
                max_depth=3,
                learning_rate=0.1,
                objective="multi:softprob",
                num_class=2,
                eval_metric="mlogloss",
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                random_state=42
            )
            self.xgb_model = None

    # ============================================================
    # LOAD DATA
    # ============================================================
    def load_data(self):
        self.df_evollis = pd.read_csv(self.evollis_path, sep=";")
        self.df_hybris = pd.read_csv(self.hybris_path, sep=";")

    # ============================================================
    # MERGE + FEATURES
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

        self.df["gap_evollis"] = abs(
            self.df["total_price_EVOLLIS"] - self.df["expected_price_evollis"]
        )
        self.df["gap_hybris"] = abs(
            self.df["total_price_HYBRIS"] - self.df["expected_price_hybris"]
        )

        self.df["is_incoherent"] = (
            (self.df["gap_evollis"] > 0) | (self.df["gap_hybris"] > 0)
        ).astype(int)

    # ============================================================
    # TRAIN ORIGIN MODEL (🔥 VRAI WARM START)
    # ============================================================
    def train_origin_model(self):
        df_train = self.df[self.df["is_incoherent"] == 1].copy()

        # ➜ label faible (bruité mais cohérent)
        df_train["weak_origin"] = np.where(
            df_train["gap_evollis"] >= df_train["gap_hybris"],
            0,  # EVOLLIS
            1   # HYBRIS
        )

        X = df_train[self.FEATURES]
        y = df_train["weak_origin"]

        self.clf_origin.fit(
            X,
            y,
            xgb_model=self.xgb_model,  # 🔥 INDISPENSABLE
            verbose=False
        )

        # ➜ sauvegarde complète du modèle
        joblib.dump(self.clf_origin, self.model_path)

        print(
            f"🌱 Nombre total d'arbres : "
            f"{self.clf_origin.get_booster().num_boosted_rounds()}"
        )

    # ============================================================
    # PREDICTION
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
    # PIPELINE
    # ============================================================
    def run(self):
        self.load_data()
        self.prepare_data()
        self.train_origin_model()
        self.predict()
        return self.df_result


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    workshop = ETAPE5_modelML_warmStart(
        evollis_path=str(_DATA_DIR / "evollis_products_V2_01012026.csv"),
        hybris_path=str(_DATA_DIR / "hybris_products_V2_01012026.csv")
    )

    df_final = workshop.run()
    df_final.to_csv(str(Path(__file__).parent / "ETAPE5_modelML_warmStart.csv"), sep=";", index=False)

    print(df_final[[
        "order_id",
        "product_name",
        "is_incoherent",
        "p_origin_evollis",
        "p_origin_hybris",
        "predicted_origin"
    ]])