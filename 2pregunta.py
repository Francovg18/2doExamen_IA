from google.colab import drive
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

drive.mount("/content/drive")
os.chdir("/content/drive/MyDrive/2do:241")

dataset = 'iris.csv'
df = pd.read_csv(dataset)
df['species'] = df['species'].astype('category').cat.codes

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X = (X - X.mean(axis=0)) / X.std(axis=0)
num_classes = len(np.unique(y))
y_one_hot = np.eye(num_classes)[y]

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
    
    def step_function(self, x):
        return np.where(x >= 0, 1, 0)
    
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1)
        self.A1 = self.step_function(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2)
        self.A2 = self.step_function(self.Z2)
        return self.A2
    
    def backward(self, X, y, output):
        output_error = y - output
        output_delta = output_error
        
        hidden_error = np.dot(output_delta, self.W2.T)
        hidden_delta = hidden_error * self.step_function(self.Z1)
        
        self.W2 += self.learning_rate * np.dot(self.A1.T, output_delta)
        self.W1 += self.learning_rate * np.dot(X.T, hidden_delta)
    
    def train(self, X, y, epochs):
        self.losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            loss = np.mean(np.square(y - output))
            self.losses.append(loss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Pérdida: {loss}")

input_size = X.shape[1]
hidden_size = 5
output_size = num_classes
learning_rate = 0.2
epochs = 1000

nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
nn.train(X, y_one_hot, epochs)

predictions = nn.forward(X)
predicted_classes = np.argmax(predictions, axis=1)

accuracy = np.mean(predicted_classes == y)
print(f"Precisión: {accuracy}")

plt.plot(nn.losses)
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.title('Pérdida durante el entrenamiento')
plt.show()
cm = confusion_matrix(y, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=df['species'].astype('category').cat.categories)
disp.plot()
plt.title('Matriz de confusión')
plt.show()
