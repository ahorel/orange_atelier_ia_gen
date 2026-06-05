from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor

_DATA_DIR = Path(__file__).parent.parent / "data"

"""Double modèle (classification + régression)

Pipeline séquentiel

Décision → correction

Logique hybride règles + ML

Architecture interprétable"""
class ETAPE1_modelML_supervise:

    def __init__(self, evollis_path: str, hybris_path: str):
        try:
            self.evollis_path = evollis_path
            self.hybris_path = hybris_path

            self.ORIGIN_LABELS = {
                -1: "OK",
                 0: "EVOLLIS",
                 1: "HYBRIS",
                 2: "EVOLLIS & HYBRIS"
            }

            self.FEATURES = [
                "quantity",
                "unit_price_EVOLLIS",
                "unit_price_HYBRIS",
                "total_price_EVOLLIS",
                "total_price_HYBRIS"
            ]

            self.clf_origin = None
            self.reg_price = None

        except Exception as e:
            print(f"[INIT ERROR] {e}")

    # ============================================================
    # 1. CHARGEMENT DES DONNÉES
    # ============================================================
    def load_data(self):
        try:
            self.df_evollis = pd.read_csv(self.evollis_path, sep=";")
            self.df_hybris = pd.read_csv(self.hybris_path, sep=";")
        except Exception as e:
            print(f"[LOAD DATA ERROR] {e}")

    # ============================================================
    # 2. FUSION + FEATURES
    # ============================================================
    def prepare_data(self):
        try:
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

        except Exception as e:
            print(f"[PREPARE DATA ERROR] {e}")

    # ============================================================
    # 3. STATUT + ORIGINE ERREUR
    # ============================================================
    def mark_status_and_origin(self):
        try:
            def _mark(row):
                ev_ok = row["total_price_EVOLLIS"] == row["expected_price_evollis"]
                hy_ok = row["total_price_HYBRIS"] == row["expected_price_hybris"]

                if ev_ok and hy_ok:
                    return pd.Series(["OK", -1])
                if not ev_ok and hy_ok:
                    return pd.Series(["KO", 0])
                if ev_ok and not hy_ok:
                    return pd.Series(["KO", 1])
                return pd.Series(["KO", 2])

            self.df[["status", "error_origin"]] = self.df.apply(_mark, axis=1)
            self.df["error_origin_label"] = self.df["error_origin"].map(self.ORIGIN_LABELS)

        except Exception as e:
            print(f"[STATUS ERROR] {e}")

    # ============================================================
    # 4. ENTRAÎNEMENT CLASSIFIEUR ORIGINE
    # ============================================================
    def train_origin_model(self):
        try:
            df_train = self.df[self.df["status"] == "KO"]

            X = df_train[self.FEATURES]
            y = df_train["error_origin"]

            self.clf_origin = XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                objective="multi:softmax",
                num_class=3,
                random_state=42,
                eval_metric="mlogloss"
            )

            self.clf_origin.fit(X, y)

        except Exception as e:
            print(f"[TRAIN ORIGIN MODEL ERROR] {e}")

    # ============================================================
    # 5. ENTRAÎNEMENT RÉGRESSION PRIX
    # ============================================================
    def train_price_model(self):
        try:
            self.df["true_price"] = np.where(
                self.df["error_origin"] == 0,
                self.df["expected_price_evollis"],
                self.df["expected_price_hybris"]
            )

            df_train = self.df[self.df["status"] == "KO"]

            X = df_train[self.FEATURES]
            y = df_train["true_price"]

            self.reg_price = XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                objective="reg:squarederror",
                random_state=42
            )

            self.reg_price.fit(X, y)

        except Exception as e:
            print(f"[TRAIN PRICE MODEL ERROR] {e}")

    # ============================================================
    # 6. PRÉDICTION + CORRECTION
    # ============================================================
    def predict_and_correct(self):
        try:
            self.df_result = self.df.copy()
            mask_ko = self.df_result["status"] == "KO"

            self.df_result["predicted_origin"] = -1

            X_cls = self.df_result.loc[mask_ko, self.FEATURES]
            preds = self.clf_origin.predict(X_cls)

            self.df_result.loc[mask_ko, "predicted_origin"] = preds
            self.df_result["predicted_origin_label"] = (
                self.df_result["predicted_origin"].map(self.ORIGIN_LABELS)
            )

            self.df_result["corrected_price"] = self.df_result["total_price_EVOLLIS"]

            X_reg = self.df_result.loc[mask_ko, self.FEATURES]
            self.df_result.loc[mask_ko, "corrected_price"] = (
                self.reg_price.predict(X_reg)
            )

        except Exception as e:
            print(f"[PREDICTION ERROR] {e}")

    # ============================================================
    # 7. EXPLICATION MÉTIER LIGNE PAR LIGNE
    # ============================================================
    def add_scenario_explanation(self):
        try:
            def explain(row):
                if row["status"] == "OK":
                    return (
                        "Les montants EVOLLIS et HYBRIS sont cohérents avec "
                        "le prix unitaire et la quantité. Aucun correctif nécessaire."
                    )

                return (
                    f"Incohérence détectée. Le modèle identifie une erreur côté "
                    f"{row['predicted_origin_label']}. "
                    f"Le prix EVOLLIS ({row['total_price_EVOLLIS']}) et "
                    f"le prix HYBRIS ({row['total_price_HYBRIS']}) sont analysés. "
                    f"Un prix corrigé de {round(row['corrected_price'], 2)} est proposé."
                )

            self.df_result["scenario_explanation"] = self.df_result.apply(explain, axis=1)

        except Exception as e:
            print(f"[EXPLANATION ERROR] {e}")

    # ============================================================
    # 8. PIPELINE COMPLET
    # ============================================================
    def run(self):
        try:
            self.load_data()
            self.prepare_data()
            self.mark_status_and_origin()
            self.train_origin_model()
            self.train_price_model()
            self.predict_and_correct()
            self.add_scenario_explanation()
            return self.df_result
        except Exception as e:
            print(f"[PIPELINE ERROR] {e}")
            return None

if __name__ == "__main__":
    workshop = ETAPE1_modelML_supervise(
        evollis_path=str(_DATA_DIR / "evollis_products.csv"),
        hybris_path=str(_DATA_DIR / "hybris_products.csv")
    )

    df_final = workshop.run()
    df_final.to_csv(str(Path(__file__).parent / "ETAPE1_modelML_supervise.csv"), sep=";")
    print(df_final[[
        "order_id",
        "order_date",
        "product_name",
        "status",
        "error_origin_label",
        "predicted_origin_label",
        "corrected_price",
        "scenario_explanation"
    ]])