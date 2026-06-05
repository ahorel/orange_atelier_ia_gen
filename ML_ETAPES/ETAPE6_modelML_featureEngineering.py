from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import os
import joblib

_DATA_DIR   = Path(__file__).parent.parent / "data"
_MODELS_DIR = Path(__file__).parent / "models"

"""❝ Ce n'est pas le modèle qui s'est amélioré.
Ce sont les informations qu'on lui a données qui sont devenues plus riches.
Un modèle intelligent sur des features pauvres reste un modèle pauvre. ❞

Nouveautés ETAPE 6 — Feature Engineering enrichi :
  - ratio_evollis  : gap relatif EVOLLIS (% de l'erreur sur le total)
  - ratio_hybris   : gap relatif HYBRIS
  - product_label  : encodage numérique du nom produit (LabelEncoder)
  - month          : mois de la commande (saisonnalité)
  - quarter        : trimestre de la commande
"""


class ETAPE6_modelML_featureEngineering:

    def __init__(self, evollis_path: str, hybris_path: str):
        self.evollis_path = evollis_path
        self.hybris_path  = hybris_path

        self.ORIGIN_LABELS = {0: "EVOLLIS", 1: "HYBRIS"}

        # ➜ Features enrichies (ETAPE2-5 + nouvelles)
        self.FEATURES = [
            "quantity",
            "unit_price_EVOLLIS",
            "unit_price_HYBRIS",
            "total_price_EVOLLIS",
            "total_price_HYBRIS",
            "gap_evollis",
            "gap_hybris",
            # Nouvelles features ETAPE 6
            "ratio_evollis",    # gap / total_price_EVOLLIS  → % d'erreur relative
            "ratio_hybris",     # gap / total_price_HYBRIS
            "product_label",    # nom produit encodé numériquement
            "month",            # mois (1–12) → saisonnalité
            "quarter",          # trimestre (1–4)
        ]

        self.model_path       = str(_MODELS_DIR / "xgb_origin_model_v6.pkl")
        self.n_estimators_step = 50
        self.clf_origin        = None
        self.xgb_model         = None
        self.product_encoder   = LabelEncoder()

        self._load_or_init_model()

    # ============================================================
    # MODEL LOAD / INIT (warm start)
    # ============================================================
    def _load_or_init_model(self):
        if os.path.exists(self.model_path):
            print("🔁 Warm start : reprise du modèle ETAPE6 existant")
            self.clf_origin = joblib.load(self.model_path)
            self.clf_origin.n_estimators += self.n_estimators_step
            self.xgb_model = self.clf_origin.get_booster()
        else:
            print("🆕 Création initiale du modèle ETAPE6")
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
    # 1. LOAD
    # ============================================================
    def load_data(self):
        self.df_evollis = pd.read_csv(self.evollis_path, sep=";")
        self.df_hybris  = pd.read_csv(self.hybris_path,  sep=";")

    # ============================================================
    # 2. MERGE + FEATURES DE BASE
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
    # 3. FEATURE ENGINEERING ENRICHI (nouveauté ETAPE 6)
    # ============================================================
    def enrich_features(self):
        # --- Ratio relatif (% d'erreur) ---
        # Évite que les grosses commandes dominent le modèle
        self.df["ratio_evollis"] = np.where(
            self.df["total_price_EVOLLIS"] != 0,
            self.df["gap_evollis"] / self.df["total_price_EVOLLIS"],
            0.0
        )
        self.df["ratio_hybris"] = np.where(
            self.df["total_price_HYBRIS"] != 0,
            self.df["gap_hybris"] / self.df["total_price_HYBRIS"],
            0.0
        )

        # --- Encodage produit ---
        # LabelEncoder : "Clavier USB" → 0, "Souris sans fil" → 1, etc.
        # Le modèle peut apprendre que certains produits ont plus d'erreurs EVOLLIS
        self.df["product_label"] = self.product_encoder.fit_transform(
            self.df["product_name"].astype(str)
        )

        # --- Features temporelles ---
        self.df["order_date"] = pd.to_datetime(
            self.df["order_date"], dayfirst=True, errors="coerce"
        )
        self.df["month"]   = self.df["order_date"].dt.month.fillna(0).astype(int)
        self.df["quarter"] = self.df["order_date"].dt.quarter.fillna(0).astype(int)

        print(f"   ✅ {len(self.FEATURES)} features actives : {self.FEATURES}")

    # ============================================================
    # 4. ENTRAÎNEMENT ORIGINE (label faible + warm start)
    # ============================================================
    def train_origin_model(self):
        df_train = self.df[self.df["is_incoherent"] == 1].copy()

        df_train["weak_origin"] = np.where(
            df_train["gap_evollis"] >= df_train["gap_hybris"],
            0,   # EVOLLIS
            1    # HYBRIS
        )

        X = df_train[self.FEATURES]
        y = df_train["weak_origin"]

        self.clf_origin.fit(
            X,
            y,
            xgb_model=self.xgb_model,
            verbose=False
        )

        joblib.dump(self.clf_origin, self.model_path)
        print(
            f"   🌱 Arbres totaux : "
            f"{self.clf_origin.get_booster().num_boosted_rounds()}"
        )

    # ============================================================
    # 5. PRÉDICTION AVEC PROBABILITÉS
    # ============================================================
    def predict(self):
        self.df_result = self.df.copy()

        mask = self.df_result["is_incoherent"] == 1
        X    = self.df_result.loc[mask, self.FEATURES]

        probas = self.clf_origin.predict_proba(X)

        self.df_result.loc[mask, "p_origin_evollis"] = probas[:, 0]
        self.df_result.loc[mask, "p_origin_hybris"]  = probas[:, 1]

        self.df_result["predicted_origin"] = np.where(
            self.df_result["p_origin_evollis"] >= self.df_result["p_origin_hybris"],
            "EVOLLIS",
            "HYBRIS"
        )

        self.df_result[["p_origin_evollis", "p_origin_hybris"]] = (
            self.df_result[["p_origin_evollis", "p_origin_hybris"]].fillna(0)
        )

    # ============================================================
    # 6. PIPELINE COMPLET
    # ============================================================
    def run(self):
        self.load_data()
        self.prepare_data()
        self.enrich_features()
        self.train_origin_model()
        self.predict()
        return self.df_result


if __name__ == "__main__":
    workshop = ETAPE6_modelML_featureEngineering(
        evollis_path=str(_DATA_DIR / "evollis_products_V2_01012026.csv"),
        hybris_path=str(_DATA_DIR / "hybris_products_V2_01012026.csv")
    )

    df_final = workshop.run()
    df_final.to_csv(str(Path(__file__).parent / "ETAPE6_modelML_featureEngineering.csv"), sep=";", index=False)

    print(df_final[[
        "order_id",
        "product_name",
        "is_incoherent",
        "ratio_evollis",
        "ratio_hybris",
        "p_origin_evollis",
        "p_origin_hybris",
        "predicted_origin"
    ]])
