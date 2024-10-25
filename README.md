# recto-verso
Séparation des italiques

## Fonctionnalités

- Détection des traits italiques (droite et gauche)
- Prétraitement d'image (contraste, luminosité, débruitage)
- Analyse avancée des gradients
- Masques de détection multiples :
  - Masque de continuité
  - Masque d'orientation
  - Masque d'intensité
  - Masque pondéré
- Superposition paramétrable des résultats
- Interface utilisateur interactive avec prévisualisation en temps réel
- Sauvegarde/Chargement des paramètres
- Export des résultats

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-username/analyseur-etiquettes.git
cd analyseur-etiquettes

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Structure du projet

```
analyseur-etiquettes/
├── config.py           # Configuration et paramètres par défaut
├── utils.py           # Fonctions utilitaires
├── image_processing.py # Traitement d'image
├── main.py            # Interface Streamlit
├── requirements.txt   # Dépendances
├── temp/             # Fichiers temporaires
└── output/           # Résultats exportés
```

## Utilisation

```bash
# Lancer l'application
streamlit run main.py
```

1. Charger une image via l'interface
2. Ajuster les paramètres dans la barre latérale :
   - Prétraitement
   - Analyse des gradients
   - Détection des traits
   - Morphologie
3. Observer les résultats en temps réel
4. Exporter le résultat final

## Dépendances principales

- Python 3.8+
- OpenCV
- NumPy
- SciPy
- scikit-image
- Streamlit
- tqdm
- numba

## Développement

Pour contribuer au projet :

1. Forker le dépôt
2. Créer une branche pour votre fonctionnalité
   ```bash
   git checkout -b feature/nouvelle-fonctionnalite
   ```
3. Commiter vos changements
   ```bash
   git commit -am 'Ajout d'une nouvelle fonctionnalité'
   ```
4. Pousser vers la branche
   ```bash
   git push origin feature/nouvelle-fonctionnalite
   ```
5. Créer une Pull Request

## Cache et optimisation

L'application utilise un système de cache pour optimiser les performances :

- Mise en cache des résultats intermédiaires
- Invalidation sélective du cache lors des changements de paramètres
- Traitement optimisé avec numba pour les calculs intensifs

## Licence

[MIT License](LICENSE)


## Remerciements

- [Streamlit](https://streamlit.io/) pour le framework d'interface
- [OpenCV](https://opencv.org/) pour le traitement d'image
