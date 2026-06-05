from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

_DATA_DIR = Path(__file__).parent.parent / "data"

# Liste de 100 produits informatiques clean sans trou de données
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

current_date = start_date
article_index = 0

while len(data) < total_lines:
    # Sélectionner l'article actuel
    product_name = product_names[article_index % len(product_names)]
    fournisseur = np.random.choice(fournisseurs)
    # Générer une quantité initiale
    quantity = np.random.randint(1, 30)
    # Générer un prix unitaire initial
    unit_price = np.random.uniform(10, 200)
    total_price = quantity * unit_price

    # Ajouter une ligne pour cette période
    data.append({
        "order_id": len(data) + 1,
        "product_name": product_name,
        "fournisseur": fournisseur,
        "order_date": current_date.strftime("%Y-%m-%d"),
        "quantity": quantity,
        "unit_price": round(unit_price, 2),
        "total_price": round(total_price, 2)
    })

    article_index += 1

    # Si on a ajouté toutes les articles pour cette période, passer à la suivante
    if article_index % len(product_names) == 0:
        # Modifier les valeurs pour la prochaine période avec des variations aléatoires
        for i in range(len(data) - len(product_names), len(data)):
            row = data[i]
            variation_pct = np.random.uniform(0.02, 0.45)
            new_quantity = max(1, int(row['quantity'] * (1 + np.random.uniform(-variation_pct, variation_pct))))
            new_unit_price = max(1, row['unit_price'] * (1 + np.random.uniform(-variation_pct, variation_pct)))
            new_total_price = new_quantity * new_unit_price
            data[i]['quantity'] = new_quantity
            data[i]['unit_price'] = round(new_unit_price, 2)
            data[i]['total_price'] = round(new_total_price, 2)

        # Passer à la prochaine date
        current_date += timedelta(days=15)

# Conversion en DataFrame et sauvegarde avec séparateur ';'
df = pd.DataFrame(data)
df.to_csv(str(_DATA_DIR / "hybris_products.csv"), sep=';', index=False)
print("Fichier 'hybris_products.csv' généré avec succès dans data/.")
