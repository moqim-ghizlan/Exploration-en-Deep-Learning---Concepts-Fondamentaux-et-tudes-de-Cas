import pandas as pd
import numpy as np

import seaborn as sns
import random
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
sns.set(style="whitegrid")

SEED = 42

def set_seed(seed=SEED):
    """Fonction pour initialiser les graines (random seed) afin de garantir la reproductibilité des résultats."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed()


# Analyse des données
def analyse_data(df, label):
    """Fonction pour effectuer une analyse préliminaire des données : distribution des classes, statistiques descriptives, etc."""
    print(f"Nombre de lignes et colonnes : {df.shape}\n")
    print(f'Valeurs manquantes :\n {df.isnull().sum()}\n')
    print(f'Type des données :\n {df.dtypes}\n')
    print("Statistiques descriptives :")
    print(df.describe())
    
    # Distribution des labels
    print(f"\nRépartition des label : {df[label].value_counts()}\n")
    plt.figure(figsize=(8, 5))
    sns.countplot(x=label, data=df, palette='viridis')
    plt.title('Répartition des label')
    plt.xticks(rotation=45)
    plt.show()
    
    # Colonnes quantitatives (numériques)
    quantitative_cols = df.select_dtypes(include=np.number).columns
    print("\nColonnes quantitatives :")
    print(quantitative_cols)

    # Histogrammes des colonnes quantitatives
    df[quantitative_cols].hist(bins=15, figsize=(15, 10), color='blue', alpha=0.7)
    plt.suptitle('Distribution des colonnes quantitatives')
    plt.show()


# Pré-traitement des données
def pretraitement(df, label, only_num=True):
    """Fonction pour effectuer le pré-traitement des données : sélection des caractéristiques et transformation des labels."""
    
    # Si on garde seulement les données numériques
    if only_num:
        classes = df[label].unique()  # Récupérer les classes uniques du label
        
        # Transformer les labels en valeurs numériques
        y = df[label].replace(classes, [0,1,2,3,4])  # Transformation des classes en valeurs numériques
        X = df.select_dtypes(include=np.number).values  # Sélectionner que les colonnes numériques
        
    else:
        # Pour les données textuelles (embeddings avec un modèle de type SentenceTransformer)
        model = SentenceTransformer("all-MiniLM-L6-v2")  # Utilisation d'un modèle pré-entrainé pour générer des embeddings de phrases
        df_copy = df.copy()  # Copier le dataframe pour éviter de modifier l'original
        classes = df_copy[label].unique()
        y = df_copy[label].replace(classes, [0,1,2,3,4])  # Transformation des classes
        df_copy.pop(label)  # Supprimer la colonne 'label' pour ne garder que les features
        
        # Sélectionner les caractéristiques numériques
        X_num = df_copy.select_dtypes(include=np.number).values
        # Sélectionner les caractéristiques catégorielles
        col = df_copy.select_dtypes(include=pd.Categorical).columns
        # Embedding des colonnes catégorielles avec le modèle de sentence-transformer
        embeddings = model.encode(df_copy[col].values)
        X_cat_transformer = embeddings
        
        # Concaténer les données numériques et les embeddings
        X = np.concatenate((X_num, X_cat_transformer), axis=1)
        
    # Conversion en tensors PyTorch
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)

    # Split des données en ensembles d'entraînement et de test (70-30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
        
    return X_train, X_test, y_train, y_test


# Création d'un Dataset personnalisé pour PyTorch
class CustomDataset(Dataset):
    """Classe pour encapsuler les données dans un format compatible avec PyTorch DataLoader."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        """Retourner la taille du dataset."""
        return len(self.x)

    def __getitem__(self, idx):
        """Retourner un échantillon de données à un index donné."""
        return self.x[idx], self.y[idx]

def get_dataloader(dataset, batch_size, shuffle=True):
    """Fonction pour obtenir un DataLoader PyTorch à partir d'un Dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Evaluation des prédictions
def evaluate_pred(y_true, y_pred):
    """Fonction pour afficher les métriques d'évaluation : précision, recall, f1-score, etc."""
    print(classification_report(y_true, y_pred))
    
    print(f'Confusion matrix : \n')
    print(confusion_matrix(y_true, y_pred))
    
    print(f'\nAccuracy : {accuracy_score(y_true, y_pred)}\n')
    
    print(f"F1_score_macro : {f1_score(y_true, y_pred, average='macro')}")
    print(f"Recall : {recall_score(y_true, y_pred, average='macro')}")
    print(f"Precision : {precision_score(y_true, y_pred, average='macro')}")

# Entraînement des modèles de machine learning
def machine_learning_models(model, X_train, X_test, y_train, y_test):
    """Fonction pour entraîner et évaluer les modèles classiques de machine learning (ex. RandomForest, SVC, etc.)."""
    for name, model in model.items():
        model.fit(X_train, y_train)  # Entraîner le modèle
        pred = model.predict(X_test)  # Effectuer des prédictions
        
        print(f'Model : {name}\n')
        evaluate_pred(y_test, pred)  # Evaluer les résultats avec les métriques


# Modèle de base de réseau de neurones (MLP)
class MLP_BASE(nn.Module):
    """Modèle simple de Perceptron Multicouche (MLP) avec une couche cachée."""
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_BASE, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Couche d'entrée -> couche cachée
        self.relu = nn.ReLU()  # Activation ReLU
        self.fc2 = nn.Linear(hidden_size, output_size)  # Couche cachée -> couche de sortie
        self.softmax = nn.Softmax(dim=1)  # Softmax pour la classification multiclasse

    def forward(self, x):
        x = self.fc1(x)  # Passer dans la couche d'entrée
        x = self.relu(x)  # Appliquer la fonction d'activation ReLU
        x = self.fc2(x)  # Passer dans la couche de sortie
        x = self.softmax(x)  # Appliquer la fonction Softmax pour obtenir les probabilités
        return x


# Modèle de réseau de neurones modifié (Dropout, GELU, BatchNorm)
class MLP_MODIF(nn.Module):
    """Modèle de Perceptron Multicouche avec des techniques d'optimisation avancées (dropout, GELU, BatchNorm)."""
    def __init__(self, input_size, num_classes):
        super(MLP_MODIF, self).__init__()

        # Couche 1
        self.fc1 = nn.Linear(input_size, 32)
        self.dropout1 = nn.Dropout(0.2)  # Dropout pour éviter le sur-apprentissage
        self.act1 = nn.GELU()  # Activation GELU
        
        # Couche 2
        self.fc2 = nn.Linear(32, 10)
        self.norm = nn.BatchNorm1d(10)  # Normalisation de lot (BatchNorm)
        self.act2 = nn.GELU()
        
        # Couche 3
        self.fc3 = nn.Linear(10, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.act3 = nn.GELU()
        
        # Couche de sortie
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.act1(self.dropout1(self.fc1(x)))  # Passer à travers les couches et activation
        x = self.act2(self.norm(self.fc2(x)))
        x = self.act3(self.dropout2(self.fc3(x)))
        x = self.fc4(x)  # Retourner la sortie
        return x
    

# Entrainement du model de deep learning
def train_dl(epochs, model ,dataloader_train, input_size, lstm=False):
    """Fonction pour entraîner un modèle de deep learning en utilisant l'optimisation Adam et la CrossEntropyLoss."""


    # Fonction d'erreur : CrossEntropyLoss pour la classification multiclasse
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)   

    # Listes pour stocker les pertes
    train_losses = []

    for epoch in range(epochs):
        total_loss = 0.0  # Total loss pour l'époque

        for batch_idx, (data, target) in enumerate(dataloader_train):
            data = data.type(torch.float32)
            target = target.type(torch.long)

            if lstm:
                data = data.view(data.size(0), 1, input_size)  # (batch_size, seq_length, input_size)

                optimizer.zero_grad()   ## réinitialiser les gradients

                prediction = model(data)
            else :
                optimizer.zero_grad()

                prediction = model(data).squeeze()   ## forward pass

            loss = criterion(prediction, target) ## calcul de la fonction de coût courante
            loss.backward()  ## backpropagation pass à travers le réseau

            torch.nn.utils.clip_grad_norm_(model.parameters(), input_size)

            optimizer.step() ## mise à jour des paramètres du réseau ( w = w -lr * w.grad) équivalent à une itération du SGD
            total_loss += loss.item()  # Accumuler la perte pour cette époque

        train_losses.append(total_loss / len(dataloader_train))  # Moyenne de la perte pour cette époque

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Époque {epoch}/{epochs}: Loss = {total_loss / len(dataloader_train):.4f}")

    # Tracer les courbes d'apprentissage
    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs), train_losses, label="Loss Entraînement")
    plt.xlabel("Époque")
    plt.ylabel("Perte (Loss)")
    plt.title("Courbe d'apprentissage")
    plt.legend()
    plt.grid(True)
    plt.show()           

    return model

# Evaluer le model dl
def test_model(model, dataloader_test, input_size=397, lstm=False):
    """Fonction pour tester un modèle de deep learning et évaluer ses performances sur un ensemble de test."""

    model.eval()  # Mettre le modèle en mode évaluation (désactive Dropout, BatchNorm)
    y_true = []
    y_pred = []

    with torch.no_grad():  # Pas besoin de calculer les gradients pendant l'évaluation
        for data, target in dataloader_test:
            data = data.type(torch.float32) 
            target = target.type(torch.long)

            if lstm:
                data = data.view(data.size(0), 1, input_size)
                prediction = model(data)

            else :
                prediction = model(data).squeeze()  # Calculer les prédictions du modèle
            predicted_classes = torch.argmax(prediction, dim=1)  # Trouver la classe prédite

            y_true.extend(target.numpy())  # Stocker les vraies étiquettes
            y_pred.extend(predicted_classes.numpy())  # Stocker les étiquettes prédites

    # Appeler la fonction d'évaluation pour afficher les métriques
    evaluate_pred(y_true, y_pred)




