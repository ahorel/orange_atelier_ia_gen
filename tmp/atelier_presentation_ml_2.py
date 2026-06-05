# ============================================================
# ATELIER MACHINE LEARNING - XGBOOST
# Détection incohérences / Origine erreur / Correction prix
# ============================================================

import pandas as pd
import numpy as np

from xgboost import XGBClassifier, XGBRegressor

# ============================================================
# 1. DONNÉES D'ENTRÉE (SIMULÉES POUR L'ATELIER)
# ============================================================

# -------- EVOLLIS --------
df_evollis = pd.DataFrame({
    "order_id": [1, 2, 3, 4, 5, 6],
    "order_date": pd.to_datetime([
        "2025-06-01", "2025-06-02", "2025-06-03",
        "2025-06-04", "2025-06-05", "2025-06-06"
    ]),
    "product_name": [
        "Clavier USB",
        "Souris sans fil",
        "Écran 24 pouces",
        "Clavier USB",
        "Casque audio",
        "Webcam HD"
    ],
    "quantity": [2, 1, 3, 2, 4, 5],
    "unit_price": [10, 20, 5, 10, 8, 7],
    "total_price": [20, 20, 15, 30, 32, 40]  # ❌ erreurs 4 et 6
})
df_evollis.to_csv("evollis_products.csv",sep=";")
df_evollis=pd.read_csv("evollis_products.csv", sep=";")

# -------- HYBRIS --------
df_hybris = pd.DataFrame({
    "order_id": [1, 2, 3, 4, 5, 6],
    "order_date": pd.to_datetime([
        "2025-06-01", "2025-06-02", "2025-06-03",
        "2025-06-04", "2025-06-05", "2025-06-06"
    ]),
    "product_name": [
        "Clavier USB",
        "Souris sans fil",
        "Écran 24 pouces",
        "Clavier USB",
        "Casque audio",
        "Webcam HD"
    ],
    "quantity": [2, 1, 3, 2, 4, 5],
    "unit_price": [10, 20, 5, 12, 8, 7],
    "total_price": [20, 20, 15, 24, 20, 35]  # ❌ erreurs 5 et 6
})
df_hybris.to_csv("hybris_products.csv",sep=";")
df_hybris=pd.read_csv("hybris_products.csv", sep=";")

# ============================================================
# 2. FUSION DES SOURCES
# ============================================================

df = df_evollis.merge(
    df_hybris,
    on=["order_id", "quantity", "product_name", "order_date"],
    suffixes=("_EVOLLIS", "_HYBRIS")
)

# Prix attendus
df["expected_price_evollis"] = df["quantity"] * df["unit_price_EVOLLIS"]
df["expected_price_hybris"] = df["quantity"] * df["unit_price_HYBRIS"]

# ============================================================
# 3. MARQUAGE OK / KO + ORIGINE ERREUR
# ============================================================

def mark_status_and_origin(row):
    ev_ok = row["total_price_EVOLLIS"] == row["expected_price_evollis"]
    hy_ok = row["total_price_HYBRIS"] == row["expected_price_hybris"]

    if ev_ok and hy_ok:
        return pd.Series(["OK", -1])
    if not ev_ok and hy_ok:
        return pd.Series(["KO", 0])  # EVOLLIS
    if ev_ok and not hy_ok:
        return pd.Series(["KO", 1])  # HYBRIS
    return pd.Series(["KO", 2])      # BOTH

df[["status", "error_origin"]] = df.apply(mark_status_and_origin, axis=1)

# ============================================================
# 4. LIBELLÉS MÉTIER (LECTURE UNIQUEMENT)
# ============================================================

ORIGIN_LABELS = {
    -1: "OK",
     0: "EVOLLIS",
     1: "HYBRIS",
     2: "EVOLLIS & HYBRIS"
}

df["error_origin_label"] = df["error_origin"].map(ORIGIN_LABELS)

# ============================================================
# 5. FEATURES ML (LES COLONNES MÉTIER NE SONT PAS UTILISÉES)
# ============================================================

FEATURES = [
    "quantity",
    "unit_price_EVOLLIS",
    "unit_price_HYBRIS",
    "total_price_EVOLLIS",
    "total_price_HYBRIS"
]

# ============================================================
# 6. MODÈLE 1 - CLASSIFICATION ORIGINE ERREUR
# ============================================================

df_train_cls = df[df["status"] == "KO"]

X_cls = df_train_cls[FEATURES]
y_cls = df_train_cls["error_origin"]

clf_origin = XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    objective="multi:softmax",
    num_class=3,
    random_state=42,
    eval_metric="mlogloss"
)

clf_origin.fit(X_cls, y_cls)

# ============================================================
# 7. MODÈLE 2 - RÉGRESSION (CORRECTION PRIX)
# ============================================================

df["true_price"] = np.where(
    df["error_origin"] == 0, df["expected_price_evollis"],
    np.where(df["error_origin"] == 1, df["expected_price_hybris"],
             df["expected_price_hybris"])
)

df_train_reg = df[df["status"] == "KO"]

X_reg = df_train_reg[FEATURES]
y_reg = df_train_reg["true_price"]

reg_price = XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    objective="reg:squarederror",
    random_state=42
)

reg_price.fit(X_reg, y_reg)

# ============================================================
# 8. PRÉDICTION + CORRECTION
# ============================================================

df_result = df.copy()
mask_ko = df_result["status"] == "KO"

# ----- Origine prédite -----
df_result["predicted_origin"] = -1

X_pred_cls = df_result.loc[mask_ko, FEATURES]
y_pred_cls = pd.Series(
    clf_origin.predict(X_pred_cls),
    index=X_pred_cls.index
)

df_result.loc[mask_ko, "predicted_origin"] = y_pred_cls
df_result["predicted_origin_label"] = df_result["predicted_origin"].map(ORIGIN_LABELS)

# ----- Prix corrigé -----
df_result["corrected_price"] = df_result["total_price_EVOLLIS"]

X_pred_reg = df_result.loc[mask_ko, FEATURES]
y_pred_reg = pd.Series(
    reg_price.predict(X_pred_reg),
    index=X_pred_reg.index
)

df_result.loc[mask_ko, "corrected_price"] = y_pred_reg

# ============================================================
# 9. RÉSULTAT FINAL (DÉMO ATELIER)
# ============================================================

print("\n===== RÉSULTAT FINAL =====\n")
print(df_result[[
    "order_id",
    "order_date",
    "product_name",
    "status",
    "error_origin_label",
    "predicted_origin_label",
    "total_price_EVOLLIS",
    "total_price_HYBRIS",
    "corrected_price"
]])

# ============================================================
# FIN
# ============================================================