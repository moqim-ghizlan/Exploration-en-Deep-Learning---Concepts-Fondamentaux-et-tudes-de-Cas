# Rétropropagation - Prédiction des Portes Logiques

## Description

Ce projet implémente un modèle de réseau neuronal conçu pour prédire les sorties des portes logiques classiques (OR, AND, XOR) en utilisant la méthode de rétropropagation. En exploitant les bibliothèques **NumPy** et **Matplotlib**, ce projet offre une exploration pratique des concepts fondamentaux du deep learning.

---

## Fonctionnalités Principales

- **Modèle Neuronal** : Prédictions binaires basées sur des portes logiques.
- **Rétropropagation** : Ajustement des poids et biais pour réduire l'erreur.
- **Visualisation** : Traçage de l'évolution du coût pendant l'entraînement.

---

## Structure du Projet

- **`main.ipynb`** : Notebook principal contenant le code source.
- **`rendu1.pdf`** : Rapport expliquant les concepts et résultats.
- **`readme.md`** : Documentation initiale.

---

## Installation et Prérequis

1. Clonez le projet :

   ```bash
   git clone <url-du-dépôt>
   cd project1
   ```

2. Installez les dépendances nécessaires :

   ```bash
   pip install numpy matplotlib
   ```

3. Ouvrez le fichier `main.ipynb` dans Jupyter Notebook pour exécuter le code.

---

## Détails du Modèle

- **Initialisation** : Les poids et biais sont générés aléatoirement.
- **Propagation Avant** : Calcul des activations jusqu'à la sortie.
- **Fonctions d'Activation** :
  - **ReLU** : Pour les couches cachées.

  - **Sigmoïde** : Pour les prédictions binaires.

- **Rétropropagation** : Optimisation basée sur la descente de gradient.
- **Visualisation** : Traçage de la courbe de coût pour chaque porte logique.

---

## Résultats

Le modèle apprend efficacement les règles des portes logiques, avec une convergence rapide illustrée par les courbes de coût.

---

## Licence

Ce projet est sous licence libre. Consultez le fichier LICENSE pour plus d'informations.
