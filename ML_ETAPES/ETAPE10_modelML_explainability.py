from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import joblib

_DATA_DIR   = Path(__file__).parent.parent / "data"
_MODELS_DIR = Path(__file__).parent / "models"

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  shap non installé. Lancer : pip install shap")
    print("   L'étape s'exécutera sans les visualisations SHAP.")

"""❝ Un modèle inexplicable est un modèle invendable.
   Le métier ne fait confiance qu'à ce qu'il comprend. ❞

Nouveautés ETAPE 10 — Explicabilité par SHAP :

  SHAP (SHapley Additive exPlanations) répond à la question :
  "Pour CETTE commande, pourquoi le modèle a dit EVOLLIS ?"

  Pour chaque prédiction, SHAP calcule la contribution de chaque feature :
    gap_evollis    → +0.42  (pousse vers EVOLLIS)
    ratio_evollis  → +0.31  (pousse vers EVOLLIS)
    gap_hybris     → -0.12  (pousse vers HYBRIS)
    product_label  → +0.04  (neutre)
    ...

  Sorties de cette étape :
    - shap_top_feature   : feature la plus décisive
    - shap_top_value     : valeur de cette feature
    - shap_explanation   : phrase en français générée automatiquement
    - shap_summary.png   : graphique global (importance + direction)
"""


class ETAPE10_modelML_explainability:

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

        self.model_path      = str(_MODELS_DIR / "xgb_origin_model_v10.pkl")
        self.n_estimators    = 100
        self.clf_origin      = None
        self.product_encoder = LabelEncoder()
        self.shap_explainer  = None
        self.shap_values_    = None

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

        X_train, self.X_ko_full, y_train, _ = train_test_split(
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
    # 4. CALCUL DES SHAP VALUES (nouveauté ETAPE 10)
    # ============================================================
    def compute_shap(self):
        """
        Calcule les SHAP values pour toutes les commandes KO.

        TreeExplainer est optimisé pour XGBoost : rapide et exact.

        shap_values shape : (n_samples, n_features, n_classes)
          → On prend la classe 0 (EVOLLIS) pour l'interprétation
          → Un SHAP positif pousse vers EVOLLIS
          → Un SHAP négatif pousse vers HYBRIS
        """
        if not SHAP_AVAILABLE:
            print("   ⚠️  SHAP non disponible, étape ignorée.")
            return

        mask   = self.df["is_incoherent"] == 1
        X_ko   = self.df.loc[mask, self.FEATURES]

        print("\n🧮 Calcul des SHAP values...")
        self.shap_explainer = shap.TreeExplainer(self.clf_origin)
        self.shap_values_   = self.shap_explainer.shap_values(X_ko)

        # shap_values_ : liste [class0_array, class1_array]
        # class0 = EVOLLIS, class1 = HYBRIS
        shap_evollis = self.shap_values_[0]  # (n_ko, n_features)

        shap_df = pd.DataFrame(
            shap_evollis,
            columns=self.FEATURES,
            index=X_ko.index
        )

        # ➜ Feature la plus décisive par ligne
        abs_shap = shap_df.abs()
        self.df.loc[mask, "shap_top_feature"] = abs_shap.idxmax(axis=1).values
        top_feat_idx = abs_shap.values.argmax(axis=1)
        self.df.loc[mask, "shap_top_value"]   = shap_evollis[
            np.arange(len(shap_evollis)), top_feat_idx
        ]

        # ➜ Importance globale
        mean_abs = abs_shap.mean().sort_values(ascending=False)
        print("\n" + "=" * 50)
        print("  IMPORTANCE GLOBALE DES FEATURES (SHAP)")
        print("=" * 50)
        for feat, val in mean_abs.items():
            bar = "█" * int(val * 100)
            print(f"  {feat:<22} {val:.4f}  {bar}")
        print("=" * 50)

        self._save_shap_plot(X_ko, shap_evollis)

    def _save_shap_plot(self, X_ko: pd.DataFrame, shap_array: np.ndarray):
        """Sauvegarde le beeswarm plot SHAP si matplotlib est disponible."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            shap.summary_plot(
                shap_array,
                X_ko,
                feature_names=self.FEATURES,
                show=False,
                plot_type="dot"
            )
            output_path = str(Path(__file__).parent / "ETAPE10_shap_summary.png")
            plt.savefig(output_path, bbox_inches="tight", dpi=120)
            plt.close()
            print(f"\n   📊 Graphique SHAP sauvegardé : {output_path}")
        except Exception as e:
            print(f"   ⚠️  Impossible de sauvegarder le graphique : {e}")

    # ============================================================
    # 5. GÉNÉRATION D'EXPLICATIONS TEXTUELLES
    # ============================================================
    def generate_explanations(self):
        """
        Génère une phrase d'explication métier pour chaque commande KO.

        Exemple :
          "Incohérence EVOLLIS (confiance 89%).
           Facteur principal : gap_evollis = 12.50
           → L'écart absolu sur le total EVOLLIS est anormalement élevé."
        """
        mask   = self.df["is_incoherent"] == 1
        X_ko   = self.df.loc[mask, self.FEATURES]
        probas = self.clf_origin.predict_proba(X_ko)

        self.df.loc[mask, "p_origin_evollis"] = probas[:, 0]
        self.df.loc[mask, "p_origin_hybris"]  = probas[:, 1]

        self.df[["p_origin_evollis", "p_origin_hybris"]] = (
            self.df[["p_origin_evollis", "p_origin_hybris"]].fillna(0)
        )

        self.df["predicted_origin"] = np.where(
            self.df["p_origin_evollis"] >= self.df["p_origin_hybris"],
            "EVOLLIS",
            "HYBRIS"
        )

        FEATURE_LABELS = {
            "gap_evollis":      "l'écart absolu EVOLLIS",
            "gap_hybris":       "l'écart absolu HYBRIS",
            "ratio_evollis":    "l'écart relatif EVOLLIS (%)",
            "ratio_hybris":     "l'écart relatif HYBRIS (%)",
            "quantity":         "la quantité commandée",
            "unit_price_EVOLLIS": "le prix unitaire EVOLLIS",
            "unit_price_HYBRIS":  "le prix unitaire HYBRIS",
            "total_price_EVOLLIS": "le total EVOLLIS",
            "total_price_HYBRIS":  "le total HYBRIS",
            "product_label":    "le type de produit",
            "month":            "le mois de la commande",
            "quarter":          "le trimestre de la commande",
        }

        def _explain(row):
            if row["is_incoherent"] == 0:
                return "Commande cohérente. Aucune intervention requise."

            origin  = row.get("predicted_origin", "?")
            p_col   = "p_origin_evollis" if origin == "EVOLLIS" else "p_origin_hybris"
            conf    = row.get(p_col, 0)
            feat    = row.get("shap_top_feature", "?")
            feat_lbl = FEATURE_LABELS.get(feat, feat)
            feat_val = round(row.get(feat, 0), 3) if feat != "?" else "?"

            return (
                f"Incohérence attribuée à {origin} "
                f"(confiance {conf:.0%}). "
                f"Facteur principal : {feat_lbl} = {feat_val}."
            )

        self.df["shap_explanation"] = self.df.apply(_explain, axis=1)

    # ============================================================
    # 6. PIPELINE COMPLET
    # ============================================================
    def run(self):
        self.load_data()
        self.prepare_data()
        self.train_origin_model()
        self.compute_shap()
        self.generate_explanations()
        return self.df

    def get_results(self) -> pd.DataFrame:
        cols = [
            "order_id", "product_name", "is_incoherent",
            "p_origin_evollis", "p_origin_hybris",
            "predicted_origin",
        ]
        if SHAP_AVAILABLE:
            cols += ["shap_top_feature", "shap_top_value"]
        cols.append("shap_explanation")
        return self.df[[c for c in cols if c in self.df.columns]]


if __name__ == "__main__":
    workshop = ETAPE10_modelML_explainability(
        evollis_path=str(_DATA_DIR / "evollis_products_V2_01012026.csv"),
        hybris_path=str(_DATA_DIR / "hybris_products_V2_01012026.csv")
    )

    df_full = workshop.run()
    df_full.to_csv(str(Path(__file__).parent / "ETAPE10_modelML_explainability.csv"), sep=";", index=False)

    print(workshop.get_results().to_string(index=False))
