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
x_gauss = np.linspace(-3, 3, 10000)
y_gauss = gaussiana(x_gauss)

# assegnazione punti a dataaset
x_train_gauss, x_test_gauss, y_train_gauss, y_test_gauss = train_test_split(x_gauss,
                                                                            y_gauss,
                                                                            test_size=0.2,
                                                                            shuffle=True)

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
    tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),  # Strato di input e primo hidden
    tf.keras.layers.Dense(10, activation='relu'),  # Secondo hidden layer
    tf.keras.layers.Dense(1)  # Strato di output
])

# fase train
lr = 0.07
gauss_mlp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mean_squared_error", metrics=["mean_squared_error"])

n_epochs = 500
batch_size = 64
history = gauss_mlp.fit(x_train_gauss,
                        y_train_gauss,
                        epochs=n_epochs,
                        batch_size=batch_size,
                        verbose=1
                        )
# fase di test
test_loss = gauss_mlp.evaluate(x_test_gauss,
                               y_test_gauss,
                               verbose=0
                               )
print(f"loss sul set di test: {test_loss}")

# plot della rete sulla funzione
y_pred_gauss = gauss_mlp.predict(x_test_gauss)
plt.scatter(x_train_gauss, gaussiana(x_train_gauss))
plt.scatter(x_test_gauss, y_pred_gauss)
plt.show()
