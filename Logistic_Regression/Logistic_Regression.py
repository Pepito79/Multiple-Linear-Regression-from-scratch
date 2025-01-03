#!/home/pepito/Documents/ML/.venv/bin/python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Load dataset
df = pd.read_csv('./Multi_Student.csv')
df = df.loc[:, ['Hours Studied', 'Previous Scores', 'Performance Index']]

# Standardize the features
scaler = StandardScaler()
df[['Hours Studied', 'Previous Scores']] = scaler.fit_transform(df[['Hours Studied', 'Previous Scores']])

# Convert the performance index to binary (1 if performance >= 50, else 0)
df['Performance Index'] = df['Performance Index'].apply(lambda x: 1 if x >= 50.0 else 0)

# Features and target
X = df[['Hours Studied', 'Previous Scores']].values
y = df['Performance Index'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Sigmoid function (used in gradient descent)
def sigmoid(z):
    return 1/(1+np.exp(-z))

# Loss function to calculate error
def loss(X, W, b, y_true):
    y_predicted = sigmoid(np.dot(X, W) + b)
    eps = 10**(-9)
    mat = y_true * np.log(y_predicted + eps) + (1 - y_true) * np.log(1 - y_predicted)
    return (-1 / len(y_true) * sum(mat))

# Gradient descent function to optimize weights and bias
def gradient_descent(X, W, b, L, y_true, epochs):
    n = len(y_true)
    for i in range(epochs):
        y_predicted = sigmoid(np.dot(X, W) + b)
        dw = (1 / n) * np.dot(X.T, (y_predicted - y_true))
        db = (1 / n) * np.sum(y_predicted - y_true)
        
        W -= L * dw
        b -= L * db

        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss(X, W, b, y_true)}")

    return W, b

# Initialize weights and bias
W = np.zeros(2)  # Size equal to the number of features (columns)
b = 0
L = 0.001  # Learning rate
epochs = 9000

# Train the model using gradient descent
W, b = gradient_descent(X_train, W, b, L, y_train, epochs)

# Print learned weights and bias
print("Learned Weights:", W)
print("Learned Bias:", b)

# Make predictions on the test set
y_predicted = sigmoid(np.dot(X_test, W) + b)
y_predicted = pd.DataFrame(y_predicted)

# Convert probabilities to binary class predictions (0 or 1)
y_pred_classes = (y_predicted >= 0.5).astype(int)

# Calculate accuracy of the model
accuracy = accuracy_score(y_test, y_pred_classes)
print("Accuracy:", accuracy)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Visualize the confusion matrix using a heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predictions')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
