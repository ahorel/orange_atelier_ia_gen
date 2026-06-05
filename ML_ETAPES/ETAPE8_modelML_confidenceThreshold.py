from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import os
import joblib

_DATA_DIR   = Path(__file__).parent.parent / "data"
_MODELS_DIR = Path(__file__).parent / "models"

"""❝ En production, une mauvaise décision prise avec confiance
   coûte moins cher qu'une bonne décision prise trop tôt.
   L'incertitude est une information métier, pas un échec. ❞

Nouveautés ETAPE 8 — Seuil de confiance (gouvernance métier) :

  RÈGLE 7 corrigée (cf. README) :
  Avant : toute prédiction KO est appliquée sans condition
  Après : la correction n'est appliquée QUE si le modèle est suffisamment sûr

  Trois décisions possibles :
    → "EVOLLIS"   si p_evollis > CONFIDENCE_THRESHOLD
    → "HYBRIS"    si p_hybris  > CONFIDENCE_THRESHOLD
    → "INCERTAIN" si max(p_evollis, p_hybris) ≤ CONFIDENCE_THRESHOLD
                  → renvoi à un expert humain
"""

CONFIDENCE_THRESHOLD = 0.75


class ETAPE8_modelML_confidenceThreshold:

    def __init__(self, evollis_path: str, hybris_path: str,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.evollis_path         = evollis_path
        self.hybris_path          = hybris_path
        self.confidence_threshold = confidence_threshold

        self.ORIGIN_LABELS = {0: "EVOLLIS", 1: "HYBRIS"}

        self.FEATURES = [
            "quantity",
            "unit_price_EVOLLIS",
            "unit_price_HYBRIS",
            "total_price_EVOLLIS",
            "total_price_HYBRIS",
            "gap_evollis",
            "gap_hybris",
            "ratio_evollis",
            "ratio_hybris",
            "product_label",
            "month",
            "quarter",
        ]

        self.model_path      = str(_MODELS_DIR / "xgb_origin_model_v8.pkl")
        self.n_estimators    = 100
        self.clf_origin      = None
        self.product_encoder = LabelEncoder()

    # ============================================================
    # 1. LOAD
    # ============================================================
    def load_data(self):
        self.df_evollis = pd.read_csv(self.evollis_path, sep=";")
        self.df_hybris  = pd.read_csv(self.hybris_path,  sep=";")

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

        self.df["gap_evollis"] = abs(
            self.df["total_price_EVOLLIS"] - self.df["expected_price_evollis"]
        )
        self.df["gap_hybris"] = abs(
            self.df["total_price_HYBRIS"] - self.df["expected_price_hybris"]
        )

        self.df["is_incoherent"] = (
            (self.df["gap_evollis"] > 0) | (self.df["gap_hybris"] > 0)
        ).astype(int)

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

        self.df["product_label"] = self.product_encoder.fit_transform(
            self.df["product_name"].astype(str)
        )

        self.df["order_date"] = pd.to_datetime(
            self.df["order_date"], dayfirst=True, errors="coerce"
        )
        self.df["month"]   = self.df["order_date"].dt.month.fillna(0).astype(int)
        self.df["quarter"] = self.df["order_date"].dt.quarter.fillna(0).astype(int)

    # ============================================================
    # 3. ENTRAÎNEMENT
    # ============================================================
    def train_origin_model(self):
        df_ko = self.df[self.df["is_incoherent"] == 1].copy()

        df_ko["weak_origin"] = np.where(
            df_ko["gap_evollis"] >= df_ko["gap_hybris"],
            0,  # EVOLLIS
            1   # HYBRIS
        )

        X = df_ko[self.FEATURES]
        y = df_ko["weak_origin"]

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.clf_origin = XGBClassifier(
            n_estimators=self.n_estimators,
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

        self.clf_origin.fit(X_train, y_train, verbose=False)
        joblib.dump(self.clf_origin, self.model_path)

    # ============================================================
    # 4. PRÉDICTION AVEC SEUIL DE CONFIANCE (nouveauté ETAPE 8)
    # ============================================================
    def predict(self):
        """
        Applique le seuil de confiance.

        Logique :
          max_proba = max(p_evollis, p_hybris)

          Si max_proba > seuil  → décision automatique (EVOLLIS ou HYBRIS)
          Sinon                 → "INCERTAIN" → renvoi expert humain

        Impact métier :
          Les cas INCERTAINS ne reçoivent aucune correction automatique.
          Ils sont listés pour révision manuelle.
          On préfère NE PAS décider plutôt que de décider mal.
        """
        self.df_result = self.df.copy()

        mask = self.df_result["is_incoherent"] == 1
        X    = self.df_result.loc[mask, self.FEATURES]

        probas = self.clf_origin.predict_proba(X)

        self.df_result.loc[mask, "p_origin_evollis"] = probas[:, 0]
        self.df_result.loc[mask, "p_origin_hybris"]  = probas[:, 1]

        self.df_result[["p_origin_evollis", "p_origin_hybris"]] = (
            self.df_result[["p_origin_evollis", "p_origin_hybris"]].fillna(0)
        )

        # ➜ Confiance maximale du modèle
        self.df_result["max_confidence"] = self.df_result[[
            "p_origin_evollis", "p_origin_hybris"
        ]].max(axis=1)

        # ➜ Décision finale avec seuil
        def _decide(row):
            if row["is_incoherent"] == 0:
                return "OK"
            if row["max_confidence"] < self.confidence_threshold:
                return "INCERTAIN"
            if row["p_origin_evollis"] >= row["p_origin_hybris"]:
                return "EVOLLIS"
            return "HYBRIS"

        self.df_result["predicted_origin"] = self.df_result.apply(_decide, axis=1)
        self.df_result["decision_applied"] = (
            self.df_result["predicted_origin"].isin(["EVOLLIS", "HYBRIS"])
        )

        self._print_governance_summary()

    def _print_governance_summary(self):
        ko_df    = self.df_result[self.df_result["is_incoherent"] == 1]
        auto     = (ko_df["predicted_origin"] != "INCERTAIN").sum()
        incert   = (ko_df["predicted_origin"] == "INCERTAIN").sum()
        total_ko = len(ko_df)

        print("\n" + "=" * 50)
        print("  RAPPORT DE GOUVERNANCE — ETAPE 8")
        print("=" * 50)
        print(f"  Seuil de confiance       : {self.confidence_threshold:.0%}")
        print(f"  Cas KO détectés          : {total_ko}")
        print(f"  Décisions automatiques   : {auto}  ({auto/total_ko:.1%})")
        print(f"  Renvoyés à l'expert      : {incert}  ({incert/total_ko:.1%})")
        print("=" * 50)

    # ============================================================
    # 5. PIPELINE COMPLET
    # ============================================================
    def run(self):
        self.load_data()
        self.prepare_data()
        self.train_origin_model()
        self.predict()
        return self.df_result


if __name__ == "__main__":
    workshop = ETAPE8_modelML_confidenceThreshold(
        evollis_path=str(_DATA_DIR / "evollis_products_V2_01012026.csv"),
        hybris_path=str(_DATA_DIR / "hybris_products_V2_01012026.csv"),
        confidence_threshold=0.75
    )

    df_final = workshop.run()
    df_final.to_csv(str(Path(__file__).parent / "ETAPE8_modelML_confidenceThreshold.csv"), sep=";", index=False)

    print(df_final[[
        "order_id",
        "product_name",
        "is_incoherent",
        "p_origin_evollis",
        "p_origin_hybris",
        "max_confidence",
        "predicted_origin",
        "decision_applied"
    ]])
