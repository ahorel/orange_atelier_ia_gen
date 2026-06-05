from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import os
import joblib

_DATA_DIR   = Path(__file__).parent.parent / "data"
_MODELS_DIR = Path(__file__).parent / "models"

"""❝ Un algorithme mal configuré, c'est comme une voiture de course
   avec les mauvais pneus : la puissance est là, mais elle ne s'exprime pas. ❞

Nouveautés ETAPE 9 — Hyperparameter Tuning :

  GridSearchCV explore systématiquement toutes les combinaisons de paramètres
  et retient celle qui maximise le F1-score en validation croisée (5 folds).

  Paramètres explorés :
    max_depth          : profondeur max des arbres  [2, 3, 4]
    learning_rate      : pas d'apprentissage        [0.05, 0.1, 0.2]
    subsample          : fraction lignes par arbre  [0.7, 0.8, 1.0]
    colsample_bytree   : fraction features par arbre [0.7, 0.8, 1.0]
    n_estimators       : nombre d'arbres            [50, 100, 200]

  Total combinaisons testées : 3 × 3 × 3 × 3 × 3 = 243
  Avec 5-fold CV : 243 × 5 = 1215 entraînements

  ⚠️  Peut prendre plusieurs minutes selon la taille du dataset.
      Réduire PARAM_GRID si nécessaire pour la démo.
"""

# Grille complète (~5 min) — réduire pour démo rapide
PARAM_GRID = {
    "max_depth":        [2, 3, 4],
    "learning_rate":    [0.05, 0.1, 0.2],
    "subsample":        [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "n_estimators":     [50, 100, 200],
}

# Grille réduite pour démo rapide (~30 secondes)
PARAM_GRID_DEMO = {
    "max_depth":        [2, 3, 4],
    "learning_rate":    [0.05, 0.1],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8],
    "n_estimators":     [50, 100],
}


class ETAPE9_modelML_hyperparamTuning:

    def __init__(self, evollis_path: str, hybris_path: str,
                 demo_mode: bool = True):
        self.evollis_path = evollis_path
        self.hybris_path  = hybris_path
        self.demo_mode    = demo_mode

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

        self.model_path      = str(_MODELS_DIR / "xgb_origin_model_v9.pkl")
        self.clf_origin      = None
        self.best_params_    = {}
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
    # 3. GRID SEARCH + ENTRAÎNEMENT FINAL (nouveauté ETAPE 9)
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

        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        param_grid = PARAM_GRID_DEMO if self.demo_mode else PARAM_GRID
        n_combinations = 1
        for v in param_grid.values():
            n_combinations *= len(v)

        print(f"\n🔍 GridSearchCV : {n_combinations} combinaisons × 5 folds "
              f"= {n_combinations * 5} entraînements")
        print("   Patience...")

        base_clf = XGBClassifier(
            objective="multi:softprob",
            num_class=2,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            verbosity=0
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            estimator=base_clf,
            param_grid=param_grid,
            scoring="f1_weighted",
            cv=cv,
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X_train, y_train)

        self.best_params_ = grid_search.best_params_
        self.clf_origin   = grid_search.best_estimator_

        joblib.dump(self.clf_origin, self.model_path)

        print("\n" + "=" * 50)
        print("  MEILLEURS HYPERPARAMÈTRES — ETAPE 9")
        print("=" * 50)
        for k, v in self.best_params_.items():
            print(f"  {k:<22} : {v}")
        print(f"\n  Score F1 CV moyen    : {grid_search.best_score_:.4f}")
        print("=" * 50)

        # Évaluation sur le test
        y_pred  = self.clf_origin.predict(self.X_test)
        y_proba = self.clf_origin.predict_proba(self.X_test)[:, 1]
        f1      = f1_score(self.y_test, y_pred, average="weighted")
        roc_auc = roc_auc_score(self.y_test, y_proba)

        print(f"\n  Performances sur jeu de TEST :")
        print(f"  F1 pondéré : {f1:.4f}")
        print(f"  ROC-AUC    : {roc_auc:.4f}")
        print(classification_report(
            self.y_test, y_pred, target_names=["EVOLLIS", "HYBRIS"]
        ))

    # ============================================================
    # 4. PRÉDICTION
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

        for k, v in self.best_params_.items():
            self.df_result[f"best_{k}"] = v

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
    workshop = ETAPE9_modelML_hyperparamTuning(
        evollis_path=str(_DATA_DIR / "evollis_products_V2_01012026.csv"),
        hybris_path=str(_DATA_DIR / "hybris_products_V2_01012026.csv"),
        demo_mode=True   # ← passer à False pour la grille complète
    )

    df_final = workshop.run()
    df_final.to_csv(str(Path(__file__).parent / "ETAPE9_modelML_hyperparamTuning.csv"), sep=";", index=False)

    print(df_final[[
        "order_id",
        "product_name",
        "is_incoherent",
        "p_origin_evollis",
        "p_origin_hybris",
        "predicted_origin"
    ]])
