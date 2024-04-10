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

itr = 0
def training_callback(weights, obj_func_eval):
        global itr
        itr += 1
        print(f"{itr} {obj_func_eval}", end=' | ')
        print('\n')

pca = PCA(n_components=2)  # for example, reduce to 2 dimensions
train_sequences_pca = pca.fit_transform(train_sequences)
test_sequences_pca = pca.fit_transform(test_sequences)
print("Train Sequences Shape after PCA:",train_sequences_pca.shape)
print("Test Sequences Shape after PCA:",test_sequences_pca.shape)
feature_dim = len(train_sequences_pca[0])


from qiskit.circuit.library import ZZFeatureMap
import datetime
prep = ZZFeatureMap(feature_dim, reps=2, entanglement="full")
# prep.draw()
# prep = prep.assign_parameters(train_sequences_pca)
ansatz = RealAmplitudes(num_qubits=feature_dim, reps=4)


qc= QNNCircuit(num_qubits=feature_dim ,ansatz = ansatz, feature_map=prep)

def parity(x):
    return "{:b}".format(x).count("1") % 2
output_shape = 2

sampler_qnn = SamplerQNN(
    circuit = qc,
    interpret=parity,
    output_shape=output_shape,
    input_gradients= True
)


# def callback_graph(weights, obj_func_eval):
#     clear_output(wait=True)
#     objective_func_vals.append(obj_func_eval)
#     plt.title("Objective function value against iteration")
#     plt.xlabel("Iteration")
#     plt.ylabel("Objective function value")
#     plt.plot(range(len(objective_func_vals)), objective_func_vals)
#     plt.show()

sampler_classifier = NeuralNetworkClassifier(
    neural_network=sampler_qnn, optimizer=COBYLA(maxiter=100), callback=training_callback
)

# fit classifier to data
sampler_classifier.fit(train_sequences_pca,train_labels)

# score classifier
# Get the current date and time
current_datetime = datetime.datetime.now()

# Create a log entry with the current date, time, and score
log_entry = f"{current_datetime}: Score - {sampler_classifier.score(test_sequences_pca,test_labels)}"
print(log_entry)
# Specify the path to the log file
log_file_path = "/d:/PhD/QML_genomic/QML_genom/log.txt"

# Check if the log file already exists
if os.path.exists(log_file_path):
    # If the log file exists, append the log entry to a new line
    with open(log_file_path, "a") as log_file:
        log_file.write(f"\n{log_entry}")
else:
    # If the log file does not exist, create a new log file and write the log entry
    with open(log_file_path, "w") as log_file:
        log_file.write(log_entry)