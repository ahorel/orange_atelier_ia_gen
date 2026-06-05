README – Atelier Machine Learning XGBoost
Objectif

Détecter incohérences sur les prix de commandes entre EVOLLIS et HYBRIS, identifier l’origine de l’erreur et proposer un prix corrigé.

1. Installer Python

Télécharger Python 3.12+ : https://www.python.org/downloads/

Pendant l’installation : cocher “Add Python to PATH” puis cliquer sur Install Now

Vérifier l’installation :

python --version

2. Installer VS Code

Télécharger : https://code.visualstudio.com/

Ouvrir VS Code

Installer l’extension Python (Microsoft) depuis le Marketplace

Configurer l’interpréteur Python :

Ctrl+Shift+P → “Python: Select Interpreter” → choisir l’interpréteur installé (ex: Python 3.12)

3. Installer les librairies Python

Ouvrir un terminal ou PowerShell et exécuter :

pip install pandas numpy xgboost

4. Préparer les fichiers CSV

evollis_products.csv et hybris_products.csv avec colonnes :

order_id;quantity;product_name;order_date;unit_price;total_price
1;2;Produit A;2025-12-10;10;20
2;1;Produit B;2025-12-10;20;20


Séparateur : ;

Date format : YYYY-MM-DD

5. Exécuter le script
Option 1 – VS Code

Ouvrir le dossier du projet

Ouvrir atelier_presentation_ml.py

Cliquer sur Run ou appuyer sur F5

Option 2 – Terminal

Se placer dans le dossier du projet :

cd C:\chemin\vers\atelier_presentation_ml


Lancer le script :

python atelier_presentation_ml.py



Ligne 0 — Commande 1 (Clavier USB)
Données

Quantité = 2

EVOLLIS = 2 × 10 = 20

HYBRIS = 2 × 10 = 20

Réalité métier

EVOLLIS : OK

HYBRIS : OK

status = OK

error_origin = -1

Prédiction ML

predicted_origin = -1 (OK)

corrected_price = 20

🎯 Scénario

S1 — Cas parfaitement cohérent (aucune action)

✔ Pas de ML
✔ Pas de correction
✔ Aucun risque

🟢 Ligne 1 — Commande 2 (Souris sans fil)

Même logique que la ligne 0.

🎯 Scénario

S1 — Cas parfaitement cohérent (aucune action)

✔ Détection correcte
✔ ML inactif

🟢 Ligne 2 — Commande 3 (Écran 24 pouces)

Même logique.

🎯 Scénario

S1 — Cas parfaitement cohérent (aucune action)

📌 Ces 3 lignes illustrent pourquoi le ML ne doit PAS intervenir sur les OK.

🔵 Ligne 3 — Commande 4 (Clavier USB)
Données

Quantité = 2

EVOLLIS = 30 (❌ devrait être 20)

HYBRIS = 24 (❌ devrait être 20)

Réalité métier

EVOLLIS : KO

HYBRIS : OK par rapport à sa logique interne

error_origin = 0 (EVOLLIS)

true_price = 20

Prédiction ML

predicted_origin = EVOLLIS

corrected_price ≈ 20.05

🎯 Scénario

M1 — Vrai positif (bonne origine + bonne correction)

✔ Erreur bien attribuée
✔ Correction proche du vrai prix
✔ Bruit normal de régression

🔴 Ligne 4 — Commande 5 (Casque audio)
Données

Quantité = 4

EVOLLIS = 32 → cohérent (4 × 8)

HYBRIS = 20 → ❌ incohérent

Réalité métier

EVOLLIS : OK

HYBRIS : KO

error_origin = 1 (HYBRIS)

true_price = 32

Prédiction ML

predicted_origin = EVOLLIS ❌

corrected_price ≈ 31.99

🎯 Scénario
❌ M5 — Faux positif (mauvaise attribution de l’erreur)

👉 Erreur réelle HYBRIS
👉 Erreur attribuée à EVOLLIS

Impact

Correction “crédible”

MAIS basée sur la mauvaise source

Risque métier réel en production

📌 C’est LE faux positif typique à montrer en atelier

🔵 Ligne 5 — Commande 6 (Webcam HD)
Données

Quantité = 5

EVOLLIS = 40 (❌ devrait être 35)

HYBRIS = 35 (OK)

Réalité métier

EVOLLIS : KO

HYBRIS : OK

error_origin = 0 (EVOLLIS)

true_price = 35

Prédiction ML

predicted_origin = EVOLLIS

corrected_price ≈ 34.96

🎯 Scénario

M1 — Vrai positif (bonne origine, correction bruitée)

✔ Bonne source
✔ Correction correcte
✔ Bruit acceptable

🧾 Tableau récapitulatif clair (slide ready)
Ligne	Réalité	Prédiction	Scénario
0	OK	OK	S1 – Aucun ML
1	OK	OK	S1 – Aucun ML
2	OK	OK	S1 – Aucun ML
3	EVOLLIS KO	EVOLLIS	M1 – Vrai positif
4	HYBRIS KO	EVOLLIS	❌ Faux positif
5	EVOLLIS KO	EVOLLIS	M1 – Vrai positif






------------------------
REGLES
------------------------
RÈGLE 1 — Définition de la cohérence EVOLLIS
ev_ok = total_price_EVOLLIS == quantity * unit_price_EVOLLIS


👉 Règle métier déterministe

Hypothèse forte :

Le total EVOLLIS doit toujours être quantité × prix unitaire.

📌 Cette règle :

décide si EVOLLIS est en erreur

est utilisée comme vérité terrain

n’est jamais apprise par le modèle

🔴 RÈGLE 2 — Définition de la cohérence HYBRIS
hy_ok = total_price_HYBRIS == quantity * unit_price_HYBRIS


Même logique que pour EVOLLIS.

👉 Deuxième vérité métier absolue

🔴 RÈGLE 3 — Détection du statut OK / KO
if ev_ok and hy_ok:
    status = "OK"
else:
    status = "KO"


👉 Le ML ne détecte jamais l’erreur
👉 Il n’a pas le droit de dire “OK ou KO”

📌 C’est une barrière de sécurité industrielle.

🔴 RÈGLE 4 — Attribution de l’origine réelle de l’erreur (label)
if not ev_ok and hy_ok:
    error_origin = 0  # EVOLLIS
elif ev_ok and not hy_ok:
    error_origin = 1  # HYBRIS
elif not ev_ok and not hy_ok:
    error_origin = 2  # LES DEUX


👉 C’est LE point le plus important :

Le ML apprend à imiter cette règle

📌 Sans cette règle :

pas de label

pas d’apprentissage supervisé

pas de modèle

🔴 RÈGLE 5 — Sélection des données d’entraînement
df_train = self.df[self.df["status"] == "KO"]


👉 Règle stratégique :

On n’entraîne le ML que là où il y a une décision à prendre.

📌 C’est un choix humain, pas un apprentissage.

🔴 RÈGLE 6 — Définition du “vrai prix” (target de régression)
true_price = expected_price_evollis if error_origin == 0 else expected_price_hybris


👉 Règle métier extrêmement forte :

Si EVOLLIS est en erreur → HYBRIS a raison
Sinon → EVOLLIS a raison

📌 Le modèle n’apprend pas ce raisonnement, il apprend à reproduire ses conséquences numériques.

🔴 RÈGLE 7 — Application inconditionnelle de la correction
if status == "KO":
    corrected_price = model.predict(...)


👉 Pas de :

seuil de confiance

probabilité minimale

garde-fou métier

📌 En production, cette règle serait dangereuse.

🔴 RÈGLE 8 — Le scénario explicatif texte
"Incohérence détectée. Le modèle identifie une erreur côté X"


👉 Même l’explication est scriptée, pas générée.