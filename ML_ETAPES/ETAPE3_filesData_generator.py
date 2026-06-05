from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

_SCRIPT_DIR = Path(__file__).parent

# Liste de 100 produits informatiques sans accents
product_names = [
    "Laptop Gaming ROG", "SSD 1TB NVMe", "Moniteur 27\" 4K", "Clavier mecanique",
    "Souris Gaming RGB", "Carte graphique RTX 3080", "RAM 16GB DDR4", "Disque dur externe 2TB",
    "Casque audio gaming", "Imprimante laser couleur", "Webcam HD 1080p", "Enceintes Bluetooth",
    "Station d'accueil USB-C", "Powerbank 20000mAh", "Routeur Wi-Fi 6", "Switch Ethernet 24 ports",
    "Microphone cardioide", "Disque SSD portable 1TB", "Ventilateur CPU RGB", "Boitier PC ATX",
    "Alimentation 750W", "Kit de refroidissement liquide", "Cle USB 128GB", "Hub USB 4 ports",
    "Adaptateur HDMI 4K", "Ecran tactile 15\"", "Projecteur portable", "Tablette graphique",
    "Disque dur interne 3.5\"", "Carte mere Z490", "Barre de son 2.1", "Camera de securite IP",
    "Disque SSD M.2 512GB", "Support pour laptop", "Cable Ethernet CAT6", "Disque dur NAS 8TB",
    "Clavier retroeclaire", "Souris sans fil", "Moniteur ultra-large 34\"", "Enceintes de bureau",
    "Station meteo connectee", "Cle de securite USB", "Disque dur SSD 2TB", "Kit de montage PC",
    "Ventilateur PC RGB", "Cable DisplayPort", "Microphone USB professionnel", "Ecran gaming 144Hz",
    "Disque dur externe SSD 1TB", "Adaptateur USB-C vers HDMI", "Support vertical pour laptop",
    "Disque dur interne 2.5\"", "Clavier ergonomique", "Souris gamer sans fil", "Ecran 24\" Full HD",
    "Camera de conference", "Enceintes Bluetooth portables", "Disque SSD PCIe NVMe", "Hub USB-C multiport",
    "Cable d’alimentation secteur", "Disque dur externe 4TB", "Cle de cryptage USB", "Support pour moniteur",
    "Disque dur interne 3TB", "Microphone de studio", "Ecran 32\" 4K UHD", "Disque SSD portable 512GB",
    "Cable USB 3.0", "Ventilateur pour GPU", "Disque dur interne 1TB", "Clavier mecanique retroeclaire",
    "Souris optique gaming", "Ecran portable 13\"", "Camera de webcam 4K", "Disque dur externe 5TB",
    "Adaptateur Ethernet USB", "Support pour disque dur", "Disque SSD SATA 1TB", "Cle USB 64GB",
    "Microphone de streaming", "Ecran 27\" IPS", "Enceintes de gaming 5.1", "Disque dur interne 4TB",
    "Cable HDMI 2.0", "Disque SSD M.2 1TB", "Support pour casque audio", "Disque dur interne 6TB",
    "Cle de securite biometrie", "Support pour laptop avec refroidissement"
]

# Liste de fournisseurs
fournisseurs = ["TechSolutions", "DataStore", "DisplayWorld", "KeyboardPro", "MouseMaster"]

# Date de départ
start_date = datetime(2024, 1, 1)

# Nombre total de lignes à générer
total_lines = 5000

# Liste pour stocker toutes les lignes
data = []
#df = []

global current_date

def create_evollis_products_csv():
    # This function generates the evollis_products.csv file
    article_index = 0
    current_date = start_date

    while len(data) < total_lines:
        # Sélectionner l'article actuel
        product_name = product_names[article_index % len(product_names)]
        fournisseur = np.random.choice(fournisseurs)
        # Générer une quantité initiale ou laisser vide
        quantity = np.random.choice([np.random.randint(1, 50), None], p=[0.9, 0.1])
        # Générer un prix unitaire ou laisser vide
        unit_price = np.random.choice([np.random.uniform(10, 200), None], p=[0.9, 0.1])
        # Calcul du prix total si possible, sinon None
        if quantity is not None and unit_price is not None:
            total_price = quantity * unit_price
        else:
            total_price = None

        # Ajouter une ligne pour cette période
        data.append({
            "order_id": len(data) + 1,
            "product_name": product_name,
            "fournisseur": fournisseur,
            "order_date": current_date.strftime("%Y-%m-%d"),
            "quantity": quantity,
            "unit_price": round(unit_price, 2) if unit_price is not None else None,
            "total_price": round(total_price, 2) if total_price is not None else None
        })

    article_index += 1
    evollis_products_df(article_index, product_names, data)
    
pass

def create_hybris_products_csv(article_index, product_names, data):
   # Si on a ajouté toutes les articles pour cette période, passer à la suivante
    if article_index % len(product_names) == 0:
        for i in range(len(data) - len(product_names), len(data)):
            row = data[i]
            variation_pct = np.random.uniform(0.02, 0.45)
            new_quantity = max(1, int(row['quantity'] * (1 + np.random.uniform(-variation_pct, variation_pct))))
            new_unit_price = max(1, row['unit_price'] * (1 + np.random.uniform(-variation_pct, variation_pct)))
            new_total_price = new_quantity * new_unit_price
            data[i]['quantity'] = new_quantity
            data[i]['unit_price'] = round(new_unit_price, 2)
            data[i]['total_price'] = round(new_total_price, 2)

        current_date += timedelta(days=15)
pass

def evollis_products_df(article_index, product_names, data):
    # Si on a ajouté toutes les articles pour cette période, passer à la suivante
    if article_index % len(product_names) == 0:
        # Modifier les valeurs pour la prochaine période avec des variations aléatoires
        for i in range(len(data) - len(product_names), len(data)):
            row = data[i]
            variation_pct = np.random.uniform(0.02, 0.45)
            # Modifier la quantité ou la laisser vide
            if np.random.rand() < 0.1:
                new_quantity = None
            else:
                new_quantity = max(1, int(row['quantity'] * (1 + np.random.uniform(-variation_pct, variation_pct)))) if row['quantity'] is not None else None
            # Modifier le prix unitaire ou le laisser vide
            if np.random.rand() < 0.1:
                new_unit_price = None
            else:
                new_unit_price = max(1, row['unit_price'] * (1 + np.random.uniform(-variation_pct, variation_pct))) if row['unit_price'] is not None else None
            # Calcul du total si possible
            if new_quantity is not None and new_unit_price is not None:
                new_total_price = new_quantity * new_unit_price
            else:
                new_total_price = None
            data[i]['quantity'] = new_quantity
            data[i]['unit_price'] = round(new_unit_price, 2) if new_unit_price is not None else None
            data[i]['total_price'] = round(new_total_price, 2) if new_total_price is not None else None

        # Passer à la prochaine date
        current_date += timedelta(days=15)
pass

def generate_files(name_file="evollis"):
            # Conversion en DataFrame et sauvegarde avec séparateur ';'
    create_evollis_products_csv()
    if name_file == "evollis":
        df = pd.DataFrame(data)
        df.to_csv(str(_SCRIPT_DIR / "ETAPE3_evollis_products.csv"), sep=';', index=False)
        print("Fichier 'ETAPE3_evollis_products.csv' généré avec succès.")
    else:
        df = pd.DataFrame(data)
        df.to_csv(str(_SCRIPT_DIR / "ETAPE3_hybris_products.csv"), sep=';', index=False)
        print("Fichier 'ETAPE3_hybris_products.csv' généré avec succès.")
pass