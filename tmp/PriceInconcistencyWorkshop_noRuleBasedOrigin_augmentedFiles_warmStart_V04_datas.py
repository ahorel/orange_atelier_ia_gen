import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
import os
import ETAPE3_filesData_generator as gen
"""“On a remplacé une règle certaine… par une décision probabiliste observable.”"""
"""Le modèle n’a pas changé de logique.
Il n’a pas appris de nouvelles règles.
La seule chose qui a changé, c’est la quantité de données.
Et pourtant, ses décisions sont devenues plus fiables. ❞

❝ En Machine Learning,
on n’améliore pas toujours un modèle en le rendant plus complexe,
mais en lui montrant plus de cas réels. """

""" On utilise un entrainement a chaud minimal du modele xgboost qui améliore les probabilités de prédictions sur l'origine"""
class PriceInconsistencyWorkshop_noRuledBasedOrigin_augmentedFiles_warmStart:

    def __init__(self, evollis_path: str, hybris_path: str):
        self.gen = gen
        self.gen.generate_files("evollis")
        self.gen.generate_files("hybris")

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
        self.model_path = "xgb_origin_model.json"

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

    def train_origin_model(self):
        df_train = self.df[self.df["is_incoherent"] == 1].copy()

        # ➜ Label faible
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

        # 🔥 WARM START LE PLUS SIMPLE POSSIBLE
        if os.path.exists(self.model_path):
            print("🔁 Warm start : reprise du modèle existant")
            self.clf_origin.fit(X, y, xgb_model=self.model_path)
        else:
            print("🆕 Entraînement initial du modèle")
            self.clf_origin.fit(X, y)

        # Sauvegarde du modèle pour le prochain run
        self.clf_origin.save_model(self.model_path)

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
    workshop = PriceInconsistencyWorkshop_noRuledBasedOrigin_augmentedFiles_warmStart(
        evollis_path="evollis_products_V2_01012026.csv",
        hybris_path="hybris_products_V2_01012026.csv"
    )

    df_final = workshop.run()
    df_final.to_csv("resultat_analyse_commandes_with_no_rules_based_augmentedFiles_warmStart.csv", sep=";", index=False)

    print(df_final[[
        "order_id",
        "product_name",
        "is_incoherent",
        "p_origin_evollis",
        "p_origin_hybris",
        "predicted_origin"
    ]])