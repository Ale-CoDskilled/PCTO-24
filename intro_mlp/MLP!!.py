import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf


# non sai dettareee

# def gaussiana
def gaussiana(x):
    y = np.exp(-x**2)
    return y

# presa dati per asse x
x_gauss = np.linspace(-3, 3, 1000)
y_gauss = gaussiana(x_gauss)

# assegnazione punti a dataaset
x_train_gauss, x_test_gauss, y_train_gauss, y_test_gauss = train_test_split(x_gauss, y_gauss, test_size=0.2, shuffle=True)

# divisione dataset tra train e test
x_train_gauss = x_train_gauss.reshape(-1, 1)
x_test_gauss = x_test_gauss.reshape(-1, 1)
y_train_gauss = y_train_gauss.reshape(-1, 1)
y_test_gauss = y_test_gauss.reshape(-1, 1)

# plot dati train
plt.scatter(x_train_gauss, y_train_gauss)
plt.show()

# plot dati test
plt.scatter(x_test_gauss, y_test_gauss)
plt.show()

# creazione neurone
gauss_mlp = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, activation='relu', input_shape=(1,)),  # Strato di input e primo hidden
    tf.keras.layers.Dense(100, activation='relu'),  # Secondo hidden layer
    tf.keras.layers.Dense(1)  # Strato di output
])


