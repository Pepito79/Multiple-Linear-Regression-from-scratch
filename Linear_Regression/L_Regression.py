#!/home/pepito/Documents/ML/.venv/bin/python

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Charger les données
data = pd.read_csv('./StudentsPerformance[1].csv')
print(data.columns)

# Vérifiez les colonnes pour correspondre aux noms exacts
data.rename(columns=lambda x: x.strip(), inplace=True)

# Descente de gradient
def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)
    for i in range(n):
        x = points.loc[i, 'writing score']
        y = points.loc[i, 'reading score']
        
        m_gradient += (2/n) * (x * b_now + m_now * x * x - y * x)
        b_gradient += -(2/n) * (y - m_now * x - b_now)
    
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b

# Initialisation
m = 0
b = 0
L = 0.0001

# Descente de gradient
for i in range(1000):
    m, b = gradient_descent(m, b, data, L)
print(m, b)

# Visualisation
plt.figure(figsize=(10, 6))  # largeur=10, hauteur=6
plt.scatter(data['writing score'], data['reading score'], color='blue')
plt.xlabel('Writing Score')
plt.ylabel('Reading Score')
plt.title('Scatter Plot of Writing vs Reading Scores')
plt.plot(list(range(0, 100)), [m * x + b for x in range(0, 100)], color="red")
plt.show()


