from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    accuracy_score,
)
import os
import joblib

_DATA_DIR   = Path(__file__).parent.parent / "data"
_MODELS_DIR = Path(__file__).parent / "models"

"""❝ Un modèle sans métriques est une boussole sans nord.
On ne sait pas s'il s'améliore, se dégrade ou tourne en rond. ❞

Nouveautés ETAPE 7 — Métriques formelles :
  - Train / Test split stratifié (80 / 20)
  - Accuracy, F1-score pondéré, ROC-AUC
  - Classification report complet (précision, rappel, F1 par classe)
  - Matrice de confusion
  Toutes les métriques sont calculées sur le jeu de TEST uniquement.
"""


class ETAPE7_modelML_metrics:

    def __init__(self, evollis_path: str, hybris_path: str):
        self.evollis_path = evollis_path
        self.hybris_path  = hybris_path

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

        self.model_path      = str(_MODELS_DIR / "xgb_origin_model_v7.pkl")
        self.n_estimators    = 100
        self.clf_origin      = None
        self.product_encoder = LabelEncoder()
        self.metrics_        = {}

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
    # 3. TRAIN / TEST SPLIT + ENTRAÎNEMENT
    # ============================================================
    def train_origin_model(self):
        df_ko = self.df[self.df["is_incoherent"] == 1].copy()

        df_ko["weak_origin"] = np.where(
            df_ko["gap_evollis"] >= df_ko["gap_hybris"],
            0,   # EVOLLIS
            1    # HYBRIS
        )

        X = df_ko[self.FEATURES]
        y = df_ko["weak_origin"]

        # ➜ Split stratifié : les deux classes représentées dans train ET test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        print(f"   📊 Jeu d'entraînement : {len(self.X_train)} lignes")
        print(f"   📊 Jeu de test        : {len(self.X_test)} lignes")

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

        self.clf_origin.fit(self.X_train, self.y_train, verbose=False)
        joblib.dump(self.clf_origin, self.model_path)

    # ============================================================
    # 4. ÉVALUATION FORMELLE (nouveauté ETAPE 7)
    # ============================================================
    def evaluate(self):
        """
        Calcule toutes les métriques sur le jeu de TEST uniquement.

        Pourquoi le jeu de test uniquement ?
        → Si on évalue sur les données d'entraînement, le modèle
          a déjà "vu" les réponses : les métriques seraient trop optimistes.
        → Le test simule un cas réel : des commandes jamais vues.
        """
        y_pred  = self.clf_origin.predict(self.X_test)
        y_proba = self.clf_origin.predict_proba(self.X_test)[:, 1]

        acc     = accuracy_score(self.y_test, y_pred)
        f1      = f1_score(self.y_test, y_pred, average="weighted")
        roc_auc = roc_auc_score(self.y_test, y_proba)

        self.metrics_ = {
            "accuracy":  round(acc,     4),
            "f1_weighted": round(f1,    4),
            "roc_auc":   round(roc_auc, 4),
        }

        print("\n" + "=" * 50)
        print("  MÉTRIQUES DE PERFORMANCE — ETAPE 7")
        print("=" * 50)
        print(f"  Accuracy    : {acc:.4f}  (bonnes prédictions / total)")
        print(f"  F1 pondéré  : {f1:.4f}  (équilibre précision/rappel)")
        print(f"  ROC-AUC     : {roc_auc:.4f}  (capacité à séparer les classes)")
        print()

        print("  Rapport de classification :")
        print(classification_report(
            self.y_test, y_pred,
            target_names=["EVOLLIS", "HYBRIS"]
        ))

        cm = confusion_matrix(self.y_test, y_pred)
        print("  Matrice de confusion :")
        print(f"               Prédit EVOLLIS  Prédit HYBRIS")
        print(f"  Réel EVOLLIS     {cm[0][0]:>5}          {cm[0][1]:>5}")
        print(f"  Réel HYBRIS      {cm[1][0]:>5}          {cm[1][1]:>5}")
        print()
        print(f"  ➜ Vrais positifs EVOLLIS : {cm[0][0]}")
        print(f"  ➜ Faux positifs EVOLLIS  : {cm[1][0]}  ← erreurs coûteuses")
        print("=" * 50)

    # ============================================================
    # 5. PRÉDICTION COMPLÈTE
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

        # Ajout des métriques en colonnes de contexte
        for k, v in self.metrics_.items():
            self.df_result[f"model_{k}"] = v

    # ============================================================
    # 6. PIPELINE COMPLET
    # ============================================================
    def run(self):
        self.load_data()
        self.prepare_data()
        self.train_origin_model()
        self.evaluate()
        self.predict()
        return self.df_result


if __name__ == "__main__":
    workshop = ETAPE7_modelML_metrics(
        evollis_path=str(_DATA_DIR / "evollis_products_V2_01012026.csv"),
        hybris_path=str(_DATA_DIR / "hybris_products_V2_01012026.csv")
    )

    df_final = workshop.run()
    df_final.to_csv(str(Path(__file__).parent / "ETAPE7_modelML_metrics.csv"), sep=";", index=False)

    print(df_final[[
        "order_id",
        "product_name",
        "is_incoherent",
        "p_origin_evollis",
        "p_origin_hybris",
        "predicted_origin",
        "model_accuracy",
        "model_f1_weighted",
        "model_roc_auc"
    ]])
