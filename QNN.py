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

pca = PCA(n_components=8)  # for example, reduce to 2 dimensions
train_sequences_pca = pca.fit_transform(train_sequences)
test_sequences_pca = pca.fit_transform(test_sequences)
print("Train Sequences Shape after PCA:",train_sequences_pca.shape)
print("Test Sequences Shape after PCA:",test_sequences_pca.shape)
feature_dim = len(train_sequences_pca[0])


from qiskit.circuit.library import ZZFeatureMap
prep = ZZFeatureMap(feature_dim, reps=2, entanglement="full")
prep.draw()
# prep = prep.assign_parameters(train_sequences_pca)
ansatz = RealAmplitudes(num_qubits=feature_dim, reps=4)