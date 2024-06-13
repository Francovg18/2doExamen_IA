from google.colab import drive
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import random
import matplotlib.pyplot as plt

drive.mount("/content/drive")
os.chdir("/content/drive/MyDrive/2do:241")

dataset = 'iris.csv'
df = pd.read_csv(dataset)
df['species'] = df['species'].astype('category').cat.codes

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def calculate_cost(features, X_train, X_test, y_train, y_test):
    if not features:  
        features = list(range(X_train.shape[1]))
    X_train_subset = X_train[:, features]
    X_test_subset = X_test[:, features]

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_subset, y_train)

    y_pred = knn.predict(X_test_subset)
    accuracy = accuracy_score(y_test, y_pred)
    return 1 - accuracy  

def get_neighbor(current_features, num_features):
    neighbor = current_features.copy()
    feature = random.choice(range(num_features))
    if feature in neighbor:
        neighbor.remove(feature)
    else:
        neighbor.append(feature)
    return neighbor

def simulated_annealing(X_train, X_test, y_train, y_test, initial_temp, cooling_rate, num_iterations):
    num_features = X_train.shape[1]
    current_features = list(np.random.choice(range(num_features), size=num_features//2, replace=False))
    current_cost = calculate_cost(current_features, X_train, X_test, y_train, y_test)
    best_features = current_features
    best_cost = current_cost
    temp = initial_temp
    costs = [] 
    for iteration in range(num_iterations):
        neighbor_features = get_neighbor(current_features, num_features)
        neighbor_cost = calculate_cost(neighbor_features, X_train, X_test, y_train, y_test)

        if neighbor_cost < current_cost:
            current_features = neighbor_features
            current_cost = neighbor_cost
        else:
            if random.uniform(0, 1) < np.exp((current_cost - neighbor_cost) / temp):
                current_features = neighbor_features
                current_cost = neighbor_cost
        
        if current_cost < best_cost:
            best_features = current_features
            best_cost = current_cost

        temp *= cooling_rate
        costs.append(current_cost) 
    
    return best_features, best_cost, costs
initial_temp = 100
cooling_rate = 0.99
num_iterations = 1000

best_features, best_cost, costs = simulated_annealing(X_train, X_test, y_train, y_test, initial_temp, cooling_rate, num_iterations)
print("Mejores características:", best_features)
print("Mejor costo (1 - precisión):", best_cost)

plt.plot(costs)
plt.xlabel('Iteración')
plt.ylabel('Costo (1 - precisión)')
plt.title('Costo durante el Simulated Annealing')
plt.show()
