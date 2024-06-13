from google.colab import drive
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

drive.mount("/content/drive")
os.chdir("/content/drive/MyDrive/2do:241")

dataset = 'iris.csv'
df = pd.read_csv(dataset)

X = df.iloc[:, :-1].values 
y = df.iloc[:, -1].values  

encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

X = (X - X.mean(axis=0)) / X.std(axis=0)
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.3, random_state=42)

dim_entrada = X_entrenamiento.shape[1]
dim_oculta = 5  # Número de neuronas en la capa oculta
dim_salida = y_entrenamiento.shape[1]

np.random.seed(42)
W1 = np.random.randn(dim_entrada, dim_oculta)
b1 = np.zeros((1, dim_oculta))
W2 = np.random.randn(dim_oculta, dim_salida)
b2 = np.zeros((1, dim_salida))

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def derivada_sigmoide(x):
    return x * (1 - x)

tasa_aprendizaje = 0.4
epocas = 1000
perdidas = []
umbral_perdida = 0.01
paciencia = 100
mejora_minima = 1e-6
mejor_perdida = np.inf
sin_mejora_epocas = 0

for epoca in range(epocas):
    # Propagación hacia adelante
    z1 = np.dot(X_entrenamiento, W1) + b1
    a1 = sigmoide(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoide(z2)

    perdida = np.mean((y_entrenamiento - a2) ** 2)
    perdidas.append(perdida)

    if perdida < umbral_perdida:
        print(f'Deteniendo el entrenamiento en la época {epoca + 1} debido a que la pérdida es menor que el umbral de {umbral_perdida}.')
        break
    
    if mejor_perdida - perdida > mejora_minima:
        mejor_perdida = perdida
        sin_mejora_epocas = 0
    else:
        sin_mejora_epocas += 1

    if sin_mejora_epocas >= paciencia:
        print(f'Deteniendo el entrenamiento en la época {epoca + 1} debido a que no hubo mejora en las últimas {paciencia} épocas.')
        break

    d_a2 = 2 * (a2 - y_entrenamiento)
    d_z2 = d_a2 * derivada_sigmoide(a2)
    d_W2 = np.dot(a1.T, d_z2)
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)

    d_a1 = np.dot(d_z2, W2.T)
    d_z1 = d_a1 * derivada_sigmoide(a1)
    d_W1 = np.dot(X_entrenamiento.T, d_z1)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)

    W2 -= tasa_aprendizaje * d_W2
    b2 -= tasa_aprendizaje * d_b2
    W1 -= tasa_aprendizaje * d_W1
    b1 -= tasa_aprendizaje * d_b1

    if (epoca + 1) % 100 == 0:
        print(f'Época {epoca + 1}/{epocas}, Pérdida: {perdida}')

epocas_totales = epoca + 1
print(f'Épocas totales requeridas: {epocas_totales}')
import matplotlib.pyplot as plt
plt.xlabel("Época")
plt.ylabel("Magnitud de pérdida")
plt.plot(perdidas)
plt.title("Pérdida durante el entrenamiento de la red neuronal")
plt.show()
