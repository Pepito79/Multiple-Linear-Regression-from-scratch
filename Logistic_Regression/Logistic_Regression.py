#!/home/pepito/Documents/ML/.venv/bin/python

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Fonction sigmoïde
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Fonction de coût avec epsilon
def cost_function(W, X, b, y_true, epsilon=1e-10):
    m = X.shape[0]
    y_predicted = sigmoid(np.dot(X, W) + b)
    lce = -(1 / m) * np.sum(y_true * np.log(y_predicted + epsilon) + (1 - y_true) * np.log(1 - y_predicted + epsilon))
    return lce

# Calcul des gradients
def gradient(W, X, b, y_true):
    m = X.shape[0]
    y_predicted = sigmoid(np.dot(X, W) + b)
    dW = (1 / m) * np.dot(X.T, (y_predicted - y_true))
    db = (1 / m) * np.sum(y_predicted - y_true)
    return dW, db

# Descente de gradient
def gradient_descent(W, X, b, y_true, L, epochs):
    for i in range(epochs):
        dW, db = gradient(W, X, b, y_true)
        W -= L * dW
        b -= L * db

        # Affichage du coût à intervalles réguliers
        if i % 100 == 0:
            cost = cost_function(W, X, b, y_true)
            print(f"Iteration {i}, Cost: {cost:.6f}")
    return W, b

# Charger les données
df = pd.read_csv('./Multi_Student.csv')

# Transformation des données
df_dummies = pd.get_dummies(df)

# Définir les variables
var1 = 'Hours Studied'
var2 = 'Previous Scores'
output = 'Performance Index'

# Séparation des variables indépendantes (X) et dépendantes (y)
X = df_dummies[[var1, var2]].astype(float)
y = df_dummies[output].astype(float)

# Normaliser X avec StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Assurez-vous que y est binaire
y_train = (y_train >= 0.5).astype(int)
y_test = (y_test >= 0.5).astype(int)

# Initialisation des poids et du biais
W = np.zeros((X_train.shape[1], 1))
b = 0
y_train = y_train.values.reshape(-1, 1)  # Convertir en vecteur colonne

# Entraîner le modèle avec la descente de gradient
learning_rate = 0.01
epochs = 5000
W, b = gradient_descent(W, X_train, b, y_train, learning_rate, epochs)

# Prédictions sur les données de test
y_pred_prob = sigmoid(np.dot(X_test, W) + b)
y_pred = (y_pred_prob >= 50).astype(int)

# Évaluer les performances
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrécision : {accuracy:.4f}")
print(y_test)

