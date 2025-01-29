import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
# Percorsi delle cartelle contenenti i dati di training e test
train_dir = os.getcwd() + '/archive/train'  # Percorso della cartella di training
test_dir = os.getcwd() + '/archive/test'  # Percorso della cartella di test

# Impostazione delle dimensioni delle immagini
img_size = (128, 128)  # Dimensione delle immagini in input alla rete

# Preprocessing delle immagini e creazione di un generatore di immagini con Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,                # Normalizzazione dell'immagine (scalando i pixel tra 0 e 1)
    rotation_range=40,             # Rotazione casuale delle immagini
    width_shift_range=0.2,         # Traslazione orizzontale casuale
    height_shift_range=0.2,        # Traslazione verticale casuale
    shear_range=0.2,               # Trasformazione di taglio
    zoom_range=0.2,                # Zoom casuale
    horizontal_flip=True,          # Flip orizzontale casuale
    fill_mode='nearest'            # Modalità di riempimento per i pixel mancanti
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Solo normalizzazione per i dati di test

# Creazione dei generatori di immagini per il training e il test
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='binary',  # Due classi: benigni e maligni
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='binary',  # Due classi: benigni e maligni
    shuffle=False
)

# Costruzione del modello CNN
model = Sequential()

# Strato convoluzionale + BatchNormalization + MaxPooling
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(BatchNormalization())  # Normalizzazione per migliorare la convergenza
model.add(MaxPooling2D((2, 2)))

# Aggiungi altri strati convoluzionali
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

# Strato fully connected
model.add(Flatten())  # Appiattisce l'output 3D in un array 1D per la parte fully connected
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout per evitare overfitting

model.add(Dense(1, activation='sigmoid'))  # Un'uscita per la classificazione binaria

# Compilazione del modello
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',  # Adatto per classificazione binaria
              metrics=['accuracy'])

# Definizione dei callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Addestramento del modello
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=80,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size,
    callbacks=[early_stopping, reduce_lr]
)

# Valutazione del modello
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Predizioni sui dati di test
y_pred_prob = model.predict(test_generator)
y_pred = (y_pred_prob > 0.5).astype("int32")  # Soglia 0.5 per classificazione binaria

# Calcolo della confusion matrix
cm = confusion_matrix(test_generator.classes, y_pred)

# Funzione per visualizzare la confusion matrix
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()

# Etichette per la confusion matrix
labels = ['Benigni', 'Maligni']

# Visualizzazione della confusion matrix
plot_confusion_matrix(cm, labels)
# Visualizzazione dei risultati dell'addestramento
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Plot per la loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot per l'accuratezza
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
plot_training_history(history)
