import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
import cv2  # Libreria OpenCV per il caricamento e la manipolazione delle immagini (installabile con pip install opencv-python)
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# Verifica della disponibilità della GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Configurazione di TensorFlow per utilizzare la GPU se disponibile
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU disponibili: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        # Gestione degli errori in caso di problemi di configurazione
        print(e)
else:
    print("GPU non disponibile, verrà utilizzata la CPU.")

# Funzione per convertire un'immagine in un array numpy
def convert_image_to_array(image_dir, size):
    """
    Converte un'immagine da un file in un array numpy dopo averla ridimensionata.

    Args:
        image_dir (str): Percorso del file immagine.
        size (tuple): Dimensioni a cui ridimensionare l'immagine (larghezza, altezza).

    Returns:
        np.array: Array numpy che rappresenta l'immagine.
    """
    image = cv2.imread(image_dir)  # Legge l'immagine dal file
    if image is not None:
        image = cv2.resize(image, size)  # Ridimensiona l'immagine alle dimensioni specificate
        return img_to_array(image)  # Converte l'immagine in un array numpy
    else:
        return np.array([])  # Restituisce un array vuoto se l'immagine non è stata caricata correttamente


# Funzione per visualizzare un insieme di immagini in una griglia
def plot_images(images_arr, n_cols=4, labels=None):
    """
    Visualizza un insieme di immagini con opzione di aggiungere le etichette.

    Args:
        images_arr (list): Lista di array di immagini.
        n_cols (int): Numero di colonne nella griglia di immagini.
        labels (list): (Opzionale) Lista di etichette da visualizzare sopra le immagini.
    """
    n_rows = len(images_arr) // n_cols + (len(images_arr) % n_cols > 0)  # Calcola il numero di righe necessario
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))  # Crea la griglia di sottotrame
    for i, ax in enumerate(axes.flat):  # Ciclo su ogni immagine e asse della griglia
        if i < len(images_arr):
            ax.imshow(images_arr[i].astype('uint8'))  # Mostra l'immagine convertita in formato 'uint8'
            ax.axis('off')  # Nasconde gli assi
            if labels is not None:
                ax.set_title(labels[i], fontsize=18)  # Imposta il titolo dell'immagine con l'etichetta
    plt.tight_layout()  # Aggiusta automaticamente il layout per evitare sovrapposizioni
    plt.show()  # Mostra le immagini


# Funzione per caricare le immagini da una cartella
def load_data(root_folder):
    """
    Carica le immagini da una cartella specificata e le etichetta in base al nome della sottocartella.

    Args:
        root_folder (str): Percorso della cartella principale contenente le sottocartelle 'benign' e 'malignant'.

    Returns:
        tuple: Una lista di immagini e una lista di etichette.
    """
    images, labels = [[], []]  # Inizializza le liste per le immagini e le etichette
    for cat in ['benign', 'malignant']:  # Itera tra le categorie 'benign' e 'malignant'
        current_folder = root_folder + os.path.sep + cat  # Percorso della sottocartella corrente
        print('Caricando immagini nella cartella {}'.format(current_folder))  # Stampa il percorso della cartella corrente
        current_folder_images = os.listdir(current_folder)  # Lista dei file immagine nella cartella corrente
        for image in current_folder_images:  # Ciclo su ogni immagine nella cartella
            if image.endswith(".jpg") or image.endswith(".JPG"):  # Controlla se il file ha estensione .jpg o .JPG
                images.append(convert_image_to_array(current_folder + os.path.sep + image, img_size))  # Converte l'immagine in array e la aggiunge alla lista
                labels.append(cat)  # Aggiunge l'etichetta corrispondente
    return images, labels  # Restituisce le liste di immagini e etichette


# Funzione per visualizzare il training
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    train_loss = history['loss']
    train_acc = history['accuracy']
    val_loss = history['val_loss']
    val_acc = history['val_accuracy']
    epochs = range(1, len(train_loss) + 1)

    axes[0].plot(epochs, train_loss, 'r-', label='Training Loss')
    axes[0].plot(epochs, val_loss, 'b--', label='Validation Loss')
    axes[0].set_title('Andamento della Loss durante l\'addestramento')
    axes[0].set_xlabel('Epoche')
    axes[0].set_ylabel('Loss')

    axes[1].plot(epochs, train_acc, 'r-', label='Training Accuracy')
    axes[1].plot(epochs, val_acc, 'b--', label='Validation Accuracy')
    axes[1].set_title('Andamento della Loss durante l\'addestramento')
    axes[1].set_xlabel('Epoche')
    axes[1].set_ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Funzione per la matrice di confusione
def plot_confusion_matrix(true_labels, predicted_labels, classes):
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Definizione delle dimensioni delle immagini da utilizzare per il ridimensionamento
img_size = (128, 128)

# Percorsi delle cartelle contenenti i dati di training e test
train_folder = os.getcwd() + '/archive/train'  # Percorso della cartella di training
test_folder = os.getcwd() + '/archive/test'  # Percorso della cartella di test

# Carica i dati di training
tipo_dict = {'benign': 0, 'malignant': 1}  # Dizionario che associa etichette testuali a valori numerici
train_x, train_y = load_data(train_folder)  # Caricamento delle immagini di training e delle etichette
train_y_int = [tipo_dict[i] for i in train_y]  # Converte le etichette testuali in numeriche utilizzando il dizionario

# Controlla se le classi sono bilanciate
unique, counts = np.unique(train_y, return_counts=True)  # Conta il numero di immagini per ciascuna classe
print(dict(zip(unique, counts)))  # Stampa il conteggio delle immagini per ogni classe

# Visualizza un campione casuale di immagini dal dataset
n_img_plot = 8  # Numero di immagini da visualizzare
idx_to_plot = np.random.choice(len(train_x), n_img_plot)  # Seleziona indici casuali delle immagini da visualizzare
img_to_plot = [train_x[i] for i in idx_to_plot]  # Estrae le immagini corrispondenti agli indici selezionati
labels_to_plot = [train_y[i] for i in idx_to_plot]  # Estrae le etichette corrispondenti
plot_images(img_to_plot, n_img_plot, labels_to_plot)  # Visualizza le immagini selezionate con le etichette

# Carica i dati di test
test_x, test_y = load_data(test_folder)  # Caricamento delle immagini di test
test_x = np.array(test_x)  # Converte le immagini in un array numpy
test_y_int = np.array([tipo_dict[i] for i in test_y])  # Converte le etichette di test in numeriche

# Creazione di un oggetto ImageDataGenerator con data augmentation
train_aug = ImageDataGenerator(
    rescale=1./255,  # Normalizza i valori dei pixel in un intervallo [0, 1]
    rotation_range=40,  # Rotazione casuale delle immagini fino a 40 gradi
    width_shift_range=0.2,  # Traslazione orizzontale casuale fino al 20% della larghezza
    height_shift_range=0.2,  # Traslazione verticale casuale fino al 20% dell'altezza
    shear_range=0.2,  # Applicazione casuale di trasformazioni di taglio
    zoom_range=0.2,  # Zoom casuale sulle immagini
    horizontal_flip=True,  # Abilitazione del flip orizzontale
    fill_mode='nearest'  # Modalità di riempimento dei pixel mancanti
)

# Creazione di un oggetto ImageDataGenerator senza data augmentation, solo normalizzazione
data_norm = ImageDataGenerator(
    rescale=1./255  # Normalizza i valori dei pixel in un intervallo [0, 1]
)

batch_size = 64  # Dimensione del batch per il caricamento dei dati

# Generatore di dati di training con data augmentation
train_data = train_aug.flow_from_directory(
    train_folder,  # Percorso della cartella di training
    target_size=img_size,  # Ridimensiona le immagini alla dimensione specificata
    color_mode='rgb',  # Colore RGB
    class_mode='categorical',  # Etichette in formato categorico (one-hot encoding)
    batch_size=batch_size,  # Dimensione del batch
    shuffle=True,  # Abilita la miscelazione dei dati
    seed=42  # Fissa il seed per la riproducibilità
)

# Se si desidera utilizzare solo la normalizzazione senza data augmentation:
# train_data = data_norm.flow_from_directory(
#    train_folder,
#    batch_size=batch_size,
#    shuffle=True,
#    seed=42
# )

# Normalizzazione delle immagini di test (divisione per 255)
test_x /= 255  # Porta i valori dei pixel nell'intervallo [0, 1]


# CREAZIONE DEL MODELLO MLP
# mlp_model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(128, 128, 3)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(2, activation='softmax')
#  ])
# COMPILAZIONE DEL MODELLO
lr = 0.001
loss_fn = tf.keras.losses.CategoricalCrossentropy()
# mlp_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss_fn, metrics=['accuracy'])
# ADDESTRAMENTO DEL MODELLO
# history_mlp = mlp_model.fit(
#     np.array(train_x) / 255.0, to_categorical(train_y_int, num_classes=2),
#     validation_data=(test_x, to_categorical(test_y_int, num_classes=2)),
#     epochs=10,
#     verbose=1
# )
# PLOTS
# train_loss = history_mlp.history['loss']
# val_loss = history_mlp.history['val_loss']

# AZIONE SUL TEST
# y_pred_mlp_model_probab = mlp_model.predict(test_x)
# y_pred = np.argmax(y_pred_mlp_model_probab,axis = 1)


# classes = ['''benigni''' , '''maligni''']
# plot_training_history(history_mlp.history)
# plot_confusion_matrix(test_y_int, y_pred, classes)

# CREAZIONE DEL MODELLO CNN
cnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='softmax')
])
# COMPILAZIONE DEL MODELLO
cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss_fn, metrics=['accuracy'])
# ADDESTRAMENTO DEL MODELLO
cnn_history = cnn_model.fit(
    train_data,
    validation_data=(test_x, to_categorical(test_y_int, num_classes=2)),
    epochs=10,
    verbose=1
)
# PLOTS
cnn_train_loss = cnn_history.history['loss']
cnn_val_loss = cnn_history.history['val_loss']

# AZIONE SUL TEST
y_pred_cnn_model_probab = cnn_model.predict(test_x)
y_pred = np.argmax(y_pred_cnn_model_probab, axis=1)
cnn_test_loss, cnn_test_accuracy = cnn_model.evaluate(test_x, to_categorical(test_y_int, num_classes=2), verbose=0)

results = {
    "history": cnn_history.history,
    "test_loss": cnn_test_loss,
    "test_accuracy": cnn_test_accuracy,
    "true_labels": test_y_int.flatten(),
    "predicted_probabilities": y_pred,
    "true_features": test_x
}


classes = ['''benigni''', '''maligni''']
plot_training_history(cnn_history.history)
plot_confusion_matrix(test_y_int, y_pred, classes)
