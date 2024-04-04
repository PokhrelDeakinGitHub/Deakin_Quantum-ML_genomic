from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


# Load the preprocessed data
train_sequences = np.load('train_sequences.npy')
# print('train_sequences',train_sequences[:2])
train_labels = np.load('train_labels.npy')
# print('train_labels',train_labels[:2])
test_sequences = np.load('test_sequences.npy')
# print('test_sequences',test_sequences[:2])
test_labels = np.load('test_labels.npy')
# print('test_labels',test_labels[:2])

# Perform PCA on the training data

pca = PCA(0.50)
train_sequences_pca = pca.fit_transform(train_sequences)
print('train_sequences_pca',train_sequences_pca[:2])

# # Plot the PCA results
# plt.figure(figsize=(10, 5))
# plt.scatter(train_sequences_pca[:, 0], train_sequences_pca[:, 1], c=train_labels, cmap='viridis')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('PCA of Training Data')
# plt.colorbar()

# plt.show()

# # Perform PCA on the test data
# test_sequences_pca = pca.transform(test_sequences)
# print('test_sequences_pca',test_sequences_pca[:2])

# # Plot the PCA results  

# plt.figure(figsize=(10, 5))
# plt.scatter(test_sequences_pca[:, 0], test_sequences_pca[:, 1], c=test_labels, cmap='viridis')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')

# plt.title('PCA of Test Data')
# plt.colorbar()

# plt.show()

# Calculate explained variance ratio (EVR) vs number of components
explained_variance_ratio = pca.explained_variance_ratio_
num_components = np.arange(1, len(explained_variance_ratio) + 1)

# Plot EVR vs number of components
plt.figure(figsize=(10, 5))
plt.plot(num_components, np.cumsum(explained_variance_ratio), marker='o', linestyle='-', color='b')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio vs Number of Components')
plt.grid(True)
plt.show()