#!/home/pepito/Documents/ML/.venv/bin/python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Import data
df = pd.read_csv('./Multi_Student.csv')

# Transform data and remove non-numeric characters
df_dummies = pd.get_dummies(df)

# Define variables
var1 = 'Hours Studied'
var2 = 'Previous Scores'
output = 'Performance Index'

# Separate independent and dependent variables
X = df_dummies[[var1, var2]]
y = df_dummies[output]

# Convert columns to float64 before normalizing
X.loc[:, var1] = X.loc[:, var1].astype(float)
X.loc[:, var2] = X.loc[:, var2].astype(float)

# Normalize the columns var1 and var2 with StandardScaler
scaler = StandardScaler()
X[['Hours Studied', 'Previous Scores']] = scaler.fit_transform(X[['Hours Studied', 'Previous Scores']])

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Cost function
def cost(a0, a1, a2, X_train, y_train):
    m = len(X_train)
    y_pred = a0 + a1 * X_train[var1] + a2 * X_train[var2]
    cost_value = (1 / (2 * m)) * np.sum((y_pred - y_train) ** 2)  # Mean squared error cost
    return cost_value

# Gradient descent function
def gradient_descent(a0, a1, a2, L_rate, X_train, y_train):
    n = len(X_train)
    
    # Calculate gradients
    error = y_train - (a0 + a1 * X_train[var1] + a2 * X_train[var2])
    a0_grad = -np.mean(error)
    a1_grad = -np.mean(error * X_train[var1])
    a2_grad = -np.mean(error * X_train[var2])

    # Update parameters
    a0 -= L_rate * a0_grad
    a1 -= L_rate * a1_grad
    a2 -= L_rate * a2_grad

    return a0, a1, a2

# Initialize parameters
a0, a1, a2 = 0, 0, 0
L_rate = 0.001  # Learning rate
J_prev = float('inf')  # Initial cost
iterations = 0
threshold = 0.000001  # Convergence threshold
max_iterations = 10000 # Maximum number of iterations to avoid infinite loop
while iterations < max_iterations:
    # Perform gradient descent
    a0, a1, a2 = gradient_descent(a0, a1, a2, L_rate, X_train, y_train)

    # Calculate current cost
    J_curr = cost(a0, a1, a2, X_train, y_train)

    # Print the cost every 100 iterations (optional)
    if iterations % 100 == 0:
        print(f"Iteration {iterations}: Cost = {J_curr}")

    # Check for convergence
    if abs(J_prev - J_curr) < threshold:
        print(f"Converged at iteration {iterations} with cost {J_curr}")
        break

    J_prev = J_curr
    iterations += 1

# Print optimized parameters
print(f"Optimized parameters: a0 = {a0}, a1 = {a1}, a2 = {a2}")

# Calculate predictions with optimized parameters
y_pred = a0 + X_test[var1] * a1 + X_test[var2] * a2

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")

# Calculate the absolute difference
diff = abs(y_pred - y_test)
resultdf = pd.DataFrame({
    'Actual Results': y_test,
    'Predicted Results': y_pred,
    'Difference': diff
})

# Bar plot of absolute difference
plt.figure(figsize=(8, 6))
plt.bar(range(len(resultdf)), resultdf['Difference'], color='green')
plt.xlabel('Observation Index')
plt.ylabel('Absolute Difference')
plt.title('Absolute Difference Between Actual and Predicted Results')
plt.show()
