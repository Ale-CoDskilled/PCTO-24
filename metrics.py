import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score


# Plot di un'immagine e la sua probabilità predetta
def plot_immagine_prob(prob, img):
     plt.grid(False)
     plt.xticks([])
     plt.yticks([])
     plt.imshow(img, cmap='gray')
     y_pred = np.argmax(prob)
     plt.xlabel('{} ({:.2f}%)'.format(int(y_pred), 100 * np.max(prob)))


# Plot delle probabilità per ogni classe
def plot_prob(prob, y):
    plt.grid(False)
    plt.yticks([])
    plt.xticks(np.arange(10))
    prob_bar = plt.bar(np.arange(10), prob, color='grey')
    plt.ylim([0, 1])
    y_pred = np.argmax(prob)
    prob_bar[y].set_color('red')
    prob_bar[y_pred].set_color('green')


# Funzione per calcolare e stampare la precisione per classe
def print_class_accuracy(true_labels, predicted_labels, classes):
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Overall accuracy: {accuracy * 100:.2f}%\n")
    print("Accuracy per class:")
    for i, class_name in enumerate(classes):
        class_accuracy = np.mean([true_labels == predicted_labels][0][true_labels == i])
        print(f"{class_name}: {class_accuracy * 100:.2f}%")


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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

## ESEMPIO DI UTILIZZ0 (togliere i commenti alle righe seguenti per far funzionare)##

# Caricare i dati
# n_epochs = 25
# lr = 0.001
# with open(f'mlp_mnist_{n_epochs}_epochs_{lr}_lr.pkl', 'rb') as file:
#     results = pickle.load(file)

# plot_idx = np.random.choice(results['predicted_labels'].shape[0], 1)
# plt.figure(figsize=(6, 3))
# plt.subplot(1, 2, 1)
# plot_immagine_prob(
#     results['predicted_probabilities'][plot_idx].squeeze(),
#     results['true_features'][plot_idx].squeeze()
# )
# plt.subplot(1, 2, 2)
# plot_prob(
#     results['predicted_probabilities'][plot_idx].squeeze(),
#     results['true_labels'][plot_idx].squeeze()
# )
# plt.show()

# Precisione per classe
# classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # MNIST classes
# print_class_accuracy(results['true_labels'], results['predicted_labels'], classes)

# Matrice di confusione
# plot_confusion_matrix(results['true_labels'], results['predicted_labels'], classes)

# Plot dell'andamento di loss e accuracy durante il training
# plot_training_history(results['history'])