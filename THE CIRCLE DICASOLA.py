import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# n_gpus = len(tf.config.list_physical_devices('GPU'))
print("Num GPUs Disponibili: ", len(tf.config.list_physical_devices('GPU')))

x_circles, y_circles = make_circles(n_samples=1000, noise=0.05)

visualizza_circles = True
if visualizza_circles:
    # y = 0 e' rappresentato dal rosso, y = 1 e' rappresentato dal blu
    plt.scatter(x_circles[np.where(y_circles == 0)[0], 0], x_circles[np.where(y_circles == 0)[0], 1], color='red')
    plt.scatter(x_circles[np.where(y_circles == 1)[0], 0], x_circles[np.where(y_circles == 1)[0], 1], color='blue')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.tight_layout()
    plt.show()

x_train_circles, x_test_circles, y_train_circles, y_test_circles = train_test_split(x_circles,
                                                                                    y_circles,
                                                                                    test_size=0.2,
                                                                                    shuffle=True)
y_test_circles = y_test_circles.reshape(-1, 1)
y_train_circles = y_train_circles.reshape(-1, 1)

circles_mlp = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation="relu", input_shape=(2,)),  # Strato di input e primo hidden
    tf.keras.layers.Dense(10, activation="relu"),  # Secondo hidden layer
    tf.keras.layers.Dense(1, activation="sigmoid")  # Strato di output per classificazione binaria
])

# fase di compilazionbe
lr = 0.01
circles_mlp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                    loss="binary_crossentropy",
                    metrics=["accuracy"]
                    )
# fase train
n_epochs = 100
batch_size = 64
history = circles_mlp.fit(x_train_circles,
                          y_train_circles,
                          epochs=n_epochs,
                          batch_size=batch_size,
                          verbose=1
                          )
# fase di test
test_loss = circles_mlp.evaluate(x_test_circles,
                                 y_test_circles,
                                 verbose=0
                                 )

print(f"loss sul set di test: {test_loss}")
plt.scatter(n_epochs, test_loss)
plt.show()
