from metrics import *

lr = 0.001
n_epochs = 10

with open(f'cnn_mnist_{n_epochs}_epochs_{lr}_lr.pkl', 'rb') as file:
    results = pickle.load(file)

plot_idx = np.random.choice(results['predicted_labels'].shape[0], 1)
print(results['predicted_probabilities'][plot_idx])
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_immagine_prob(
    results['predicted_probabilities'][plot_idx].squeeze(),
    results['true_features'][plot_idx].squeeze()
)
plt.subplot(1, 2, 2)
plot_prob(
    results['predicted_probabilities'][plot_idx].squeeze(),
    results['true_labels'][plot_idx].squeeze()
)
plt.show()

# Precisione per classe
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # MNIST classes
print_class_accuracy(results['true_labels'], results['predicted_labels'], classes)

# Matrice di confusione
plot_confusion_matrix(results['true_labels'], results['predicted_labels'], classes)

# Plot dell'andamento di loss e accuracy durante il training
plot_training_history(results['history'])
