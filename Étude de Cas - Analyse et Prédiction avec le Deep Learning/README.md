# Étude de Cas - Analyse et Prédiction avec le Deep Learning

## Description

Ce projet explore l'utilisation du deep learning pour analyser et prédire des données à l'aide de plusieurs ensembles. Avec une approche méthodique et des outils robustes, ce projet comprend des notebooks reproductibles, un rapport complet, et des scripts utilitaires. Il met également en évidence les performances des modèles à travers des graphiques et des courbes.

---

## Contenu du Projet

- **Notebook** : `main.ipynb` contient l'ensemble du flux d'analyse et de prédiction.

- **Rapport** : `GHIZLAN_JUILLARD_Cas_etude_DL.pdf` documente les résultats et la méthodologie.

- **Données** :
  - `train.csv` et `test.csv` pour la formation et les tests.

- **Scripts** :
  - `utils.py` fournit des fonctions utilitaires pour la gestion des données et des modèles.

- **Graphiques** : Courbes de performance et visualisation des résultats dans `images/`.

- **Dependencies** : Spécifiées dans `requirements.txt`.


---

## Installation et Prérequis

1. Clonez le projet :

   ```bash
   git clone <url-du-dépôt>
   cd project2
   ```

2. Créez un environnement virtuel et activez-le :

   ```bash
   python3 -m venv env
   source env/bin/activate  # Linux/Mac
   env\Scripts\activate   # Windows
   ```

3. Installez les dépendances :

   ```bash
   pip install -r requirements.txt
   ```

---

## Lancement du Projet

1. Placez les ensembles de données dans le dossier `donnees_rendu2/` si ce n'est pas déjà fait.
2. Ouvrez `main.ipynb` dans Jupyter Notebook ou Jupyter Lab.
3. Exécutez chaque cellule pour reproduire les analyses et visualisations.

---

## Visualisations Incluses

- **Courbes de Performance** : Comparaison des performances des modèles de base et modifiés.
- **Distribution des Données** : Graphiques illustrant la répartition des labels et des caractéristiques.
- **Structures des Modèles** : Captures des modèles de base et modifiés.

---

## Résultats

Les résultats finaux, y compris les prédictions, sont enregistrés dans :

- `GHIZLAN_JUILLARD_prediction_base.csv`

- `GHIZLAN_JUILLARD_prediction_full.csv`


---

## Licence

Ce projet est sous licence libre. Consultez le fichier LICENSE pour plus d'informations.
