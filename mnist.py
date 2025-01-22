import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle


#caricamento e preprocessamento del dataset MNIST
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()
# one-hot-encoding delle etichette categoriche
y_train_mnist_enc = tf.keras.utils.to_categorical(y_train_mnist)
y_test_mnist_enc = tf.keras.utils.to_categorical(y_test_mnist)
# normalize x features trasformo tutti i numeri che possono assumere nella matrice all'interbvallo 0-1
x_train_mnist_norm = x_train_mnist.astype(np.float32)/255.0
x_test_mnist_norm = x_test_mnist.astype(np.float32)/255.0

#creazione del modello MLP
mnist_mlp = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), # converte l'immagine in un vettore
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2), #drop out evita l'overfitting cancellando connessioni tra neuroni ai fini di diminuire la potenza computazionale
  tf.keras.layers.Dense(10, activation='softmax')
])

lr = 0.001
n_epochs = 25
batch = 64
loss_fn = tf.keras.losses.CategoricalCrossentropy()
mnist_mlp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss_fn, metrics=['accuracy'])

train_model = True
if train_model:
    mnist_mlp_history = mnist_mlp.fit(x_train_mnist_norm, y_train_mnist_enc,
                                      validation_split=0.2,
                                      epochs=n_epochs,
                                      batch_size=batch,
                                      verbose=1)
loss = mnist_mlp_history.history["loss"]
val_loss = mnist_mlp_history.history["val_loss"]
plt.figure(figsize=(8, 6))
plt.plot(range(n_epochs), loss, 'r-', label='Training Loss')
plt.plot(range(n_epochs), val_loss, 'b--', label='Validation Loss')
plt.title('Andamento della Loss durante l\'addestramento')
plt.xlabel('Epoche')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

mnist_test_loss, mnist_test_accuracy = mnist_mlp.evaluate(x_test_mnist_norm, y_test_mnist_enc, verbose=0)
print('Test accuracy: ', mnist_test_accuracy)

# Predizioni sul test set
y_pred_mnsit_probabilities = mnist_mlp.predict(x_test_mnist)
y_pred_mnsit = np.argmax(y_pred_mnsit_probabilities,axis = 1)

conf_matrix = confusion_matrix(y_test_mnist,y_pred_mnsit)
ConfusionMatrixDisplay(conf_matrix, display_labels=np.arange(10)).plot(colorbar=True,cmap="viridis")
plt.title("Matrice di confusione")
plt.show()



results = {
    "history" :mnist_mlp_history.history,
    "test_loss": mnist_test_loss,
    "test_accuracy":mnist_test_accuracy,
    "treu_labels": y_test_mnist.flatten(),
    "predicted_probabilities": y_pred_mnsit_probabilities,
    "treu_features": x_test_mnist_norm

}
with open(f'mlp_minst_{n_epochs}_epochs_{lr}_lr.pkl', 'wb') as file:
    pickle.dump(results, file)