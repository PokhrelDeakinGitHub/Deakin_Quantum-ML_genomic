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

from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.neural_networks import SamplerQNN

algorithm_globals.random_seed = 42

warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")

DATASET = "demo_coding_vs_intergenomic_seqs"
BATCH_SIZE = 4
ansatz_reps = 4
max_train_iterations = 10

if not is_downloaded(DATASET):
    download_dataset(DATASET)

SEQ_PATH = Path.home() / '.genomic_benchmarks' / DATASET
CLASSES = [x.stem for x in (SEQ_PATH/'train').iterdir() if x.is_dir()]
NUM_CLASSES = len(CLASSES)
train_dset = tf.keras.preprocessing.text_dataset_from_directory(
    SEQ_PATH / 'train',
    batch_size=BATCH_SIZE,class_names=CLASSES)

vectorize_layer.adapt(train_dset.map(lambda x, y: x))
VOCAB_SIZE = len(vectorize_layer.get_vocabulary())
vectorize_layer.get_vocabulary()

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
#   print("1")
#   print(text)
  return vectorize_layer(text)-2, label


train_ds = train_dset.map(vectorize_text)


# Create the np_data_set as an empty list
np_data_set = []

for text_list, label_list in train_ds:
    # Convert text_list and label_list to numpy arrays just once for efficiency
    texts = text_list.numpy()
    labels = label_list.numpy()

    for text, label in zip(texts, labels):
        # Ensure sequence is 256 elements long, padding with -1 if necessary
        sequence_length = 256
        padded_sequence = np.pad(text, (0, max(sequence_length - len(text), 0)), constant_values=-1)

        # Create a dictionary with the sequence and label, and append it to the list
        sequence_dict = {
            "sequence": padded_sequence.tolist(),  # Convert numpy array back to list
            "label": label.tolist()                # Convert label to list if it's not already
        }
        np_data_set.append(sequence_dict)


# Split the data set into training and testing sets Note: here training dataset is splited into the training and testing
np_train_data = np_data_set[:70000]
np_test_data = np_data_set[-5000:]

train_sequences = [data_point["sequence"] for data_point in np_train_data]
train_labels = [data_point["label"] for data_point in np_train_data]
test_sequences = [data_point["sequence"] for data_point in np_test_data]
test_labels = [data_point["label"] for data_point in np_test_data]

train_sequences = np.array(train_sequences)
train_labels = np.array(train_labels)
test_sequences = np.array(test_sequences)
test_labels = np.array(test_labels)

print("Train Sequences Shape:", train_sequences.shape)
print("Train Labels Shape:", train_labels.shape)
print("Test Sequences Shape:", test_sequences.shape)
print("Test Labels Shape:", test_labels.shape)