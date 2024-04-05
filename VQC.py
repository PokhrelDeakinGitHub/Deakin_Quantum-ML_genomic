from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
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
from qiskit.primitives import Sampler
from qiskit_machine_learning.algorithms.classifiers import VQC
from functools import partial

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

pca = PCA(2)
train_sequences_pca = pca.fit_transform(train_sequences)
test_sequences_pca = pca.fit_transform(test_sequences)

feature_dim = len(train_sequences_pca[0])
print(f"Feature dimension: {feature_dim}")
num_qubits = int(np.log2(feature_dim))
print(f"Number of qubits required: {num_qubits}")


itr = 0
def training_callback(weights, obj_func_eval):
        global itr
        itr += 1
        print(f"{itr} {obj_func_eval}", end=' | ')



feature_map = RawFeatureVector(feature_dimension=feature_dim)
ansatz = RealAmplitudes(num_qubits = num_qubits, reps=4)
optimizer = COBYLA(maxiter=10)
vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=Sampler(),
        callback=partial(training_callback)
    )


print("Training Started")
start_time = time.time()
vqc.fit(train_sequences_pca, train_labels)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTraining complete. Time taken: {elapsed_time} seconds.")


print(f"SCORING MODEL")
train_score_q = vqc.score(train_sequences_pca, train_labels)
test_score_q = vqc.score(test_sequences_pca[:200], test_labels[:200])
print(f"train score with 8 components", train_score_q)
print(f"test score with 8 components", test_score_q)


# from qiskit import QuantumCircuit, Aer, transpile
# from qiskit.algorithms.optimizers import COBYLA
# from qiskit.circuit import ParameterVector
# from qiskit.datasets import load_iris
# from qiskit.opflow import I, X, Z, Operator
# from qiskit.quantum_info import Statevector

# # Load the Iris dataset
# dataset = load_iris()
# X, y = dataset['data'], dataset['target']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # Define the quantum circuit
# n_qubits = 4
# params = ParameterVector(n_qubits)
# circuit = QuantumCircuit(n_qubits)
# circuit.h(range(n_qubits))
# circuit.append(qnn.weights_to_quantum_circuit(params), range(n_qubits))

# # Define the objective function
# def objective(params):
#     # ... your objective function here ...
#     return expectation

# # Set up the backend and transpile the circuit
# backend = provider.get_backend('ibmq_16_melbourne')
# transpiled_circuit = transpile(circuit, backend)

# # Define the optimizer
# optimizer = COBYLA(maxiter=1000)

# # Train the quantum circuit
# result = VQE(expectation=objective, ansatz=circuit, optimizer=optimizer, quantum_instance=backend).compute_minimum_eigenvalue(params)

# # Evaluate the trained model on the test set
# accuracy = circuit.score(X_test, y_test)

# # Print the results
# print(f"Accuracy: {accuracy}")