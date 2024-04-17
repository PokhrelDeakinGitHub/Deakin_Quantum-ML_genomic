from pathlib import Path
import os
import sys
import tensorflow as tf
import warnings
import numpy as np
from genomic_benchmarks.loc2seq import download_dataset
from genomic_benchmarks.data_check import is_downloaded, info
from genomic_benchmarks.models.tf import vectorize_layer
from genomic_benchmarks.models.tf import get_basic_cnn_model_v0 as get_model
import matplotlib.pyplot as plt
import time
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC
from functools import partial
# from qiskit import Aer
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit.primitives import Sampler
from qiskit_machine_learning.circuit.library import QNNCircuit
import json
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from qiskit import ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from sklearn.decomposition import PCA
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.neural_networks import SamplerQNN

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit.circuit.library import ZZFeatureMap
import datetime
from tqdm import tqdm

algorithm_globals.random_seed = 42

warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")


algorithm_globals.random_seed = 42

# Load the preprocessed data
train_sequences = np.load('train_sequences.npy')
# print('train_sequences',train_sequences[:2])
train_labels = np.load('train_labels.npy')
# print('train_labels',train_labels[:2])
test_sequences = np.load('test_sequences.npy')
# print('test_sequences',test_sequences[:2])
test_labels = np.load('test_labels.npy')
# print('test_labels',test_labels[:2])

pca = PCA(n_components=4)  # for example, reduce to n dimensions
train_sequences_pca = pca.fit_transform(train_sequences)
test_sequences_pca = pca.fit_transform(test_sequences)
print("Train Sequences Shape after PCA:",train_sequences_pca.shape)
print("Test Sequences Shape after PCA:",test_sequences_pca.shape)

# Define the feature dimension or number of qubits
feature_dim = len(train_sequences_pca[0])

prep = ZZFeatureMap(feature_dim, reps=1)
ansatz = RealAmplitudes(num_qubits=feature_dim, reps=2)
qc= QNNCircuit(num_qubits=feature_dim ,ansatz = ansatz, feature_map=prep)

objective_func_vals=[]

def callback_graph(weights, obj_func_eval):
    clear_output(wait=False)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

def parity(x):
    return "{:b}".format(x).count("1") % 2
output_shape = 2

sampler_qnn = SamplerQNN(
    circuit = qc,
    interpret=parity,
    output_shape=output_shape,
    input_gradients= True
)


sampler_classifier = NeuralNetworkClassifier(
    neural_network=sampler_qnn, optimizer=COBYLA(maxiter=10), callback=callback_graph
)
# Train the classifier
num_epochs = 10
batch_size = 32

# Create a progress bar
progress_bar = tqdm(total=num_epochs, desc="Training")

for epoch in range(num_epochs):
    # Perform training for each batch
    for i in range(0, len(train_sequences_pca), batch_size):
        batch_sequences = train_sequences_pca[i:i+batch_size]
        batch_labels = train_labels[i:i+batch_size]
        
        
        # Train the classifier on the batch
        sampler_classifier.fit(batch_sequences, batch_labels)
        print(f"Epoch {epoch + 1} - Batch {i // batch_size + 1} completed")
    # Update the progress bar
    progress_bar.update(1)

# Close the progress bar
progress_bar.close()

from tqdm import tqdm
total_time = 0
history = {
#   "train_loss": [],
#   "test_loss": [],
  "train_accuracy": [],
  "test_accuracy": [],
}
for epoch in range(10):
  start = time.time()

  for epoch in range(num_epochs):# Perform training for each batch
    for i in range(0, len(train_sequences_pca), batch_size):
        batch_sequences = train_sequences_pca[i:i+batch_size]
        batch_labels = train_labels[i:i+batch_size]
        # Perform training for the batch
        pegasos_qsvc.fit(batch_sequences, batch_labels)
        
    end = time.time()
    total_time += end - start
    print(f"Epoch {epoch+1}/{10}: Training time: {end - start:.3f} seconds")



    # Evaluate the model on the training and test sets
    train_accuracy = pegasos_qsvc.score(train_sequences_pca, train_labels)
    test_accuracy = pegasos_qsvc.score(test_sequences_pca, test_labels)

    # Update the history dictionary

    history["train_accuracy"].append(train_accuracy)
    history["test_accuracy"].append(test_accuracy)

    # history["train_loss"].append(train_loss)
    # history["test_loss"].append(test_loss)

    print(f"Epoch {epoch+1}/{10}: Train Accuracy: {train_accuracy:.3f}, Test Accuracy: {test_accuracy:.3f}")

    log_file_path = "log_zfeatureMap.txt"
    # Check if the log file already exists
    if os.path.exists(log_file_path):
        # If the log file exists, append the log entry to a new line
        with open(log_file_path, "a") as log_file:
            log_file.write(f"\n{history}")
    else:
        # If the log file does not exist, create a new log file and write the log entry
        with open(log_file_path, "w") as log_file:
            log_file.write(history)



    # Plot the training and test loss
    clear_output(wait=True)
    # Plot the training and test accuracy

    plt.figure(figsize=(12, 6))
    plt.plot(history["train_accuracy"], label="Train Accuracy")
    plt.plot(history["test_accuracy"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()