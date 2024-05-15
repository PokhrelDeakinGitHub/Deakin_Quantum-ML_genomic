# Quantum Machine Learning for Genomic Sequence Classification

This repository is dedicated to an exploratory project that combines the cutting-edge fields of quantum computing and machine learning, with a specific focus on genomic sequence data. Our aim is to leverage the unique capabilities of quantum algorithms to improve classification accuracy in the context of genomic data, which presents its own set of challenges and opportunities for innovation.

## Project Overview

The core of this project involves the implementation and comparative analysis of various quantum machine learning algorithms, namely:

- **Variational Quantum Classifier (VQC):** Utilizes parameterized quantum circuits as a model, which are optimized to perform classification tasks.
- **Quantum Support Vector Classifier (QSVC):** A quantum-enhanced version of the classical support vector machine, designed to exploit quantum computing's ability to handle complex computations more efficiently.
- **Pegasos Quantum Support Vector Classifier:** An adaptation of the classical Pegasos (Primal Estimated sub-GrAdient SOlver for SVM) algorithm to the quantum computing domain, aiming for faster convergence on quantum hardware.
- **Quantum Neural Networks (QNNs):** An approach to neural networks where quantum circuits are used to perform nonlinear operations, potentially offering advantages in processing complex patterns in data.

### Objectives

The primary objectives of this project are to:

1. **Implement and Evaluate Quantum Machine Learning Algorithms:** To apply the aforementioned quantum algorithms on genomic sequence datasets, evaluating their performance in classification tasks.
2. **Analyze the Impact of Algorithmic and Data Encoding Parameters:** To systematically vary the parameters within each quantum machine learning algorithm and the techniques used for encoding genomic data into quantum states. This analysis aims to uncover insights into how different settings affect classification accuracy.


### Dataset

The project utilizes genomic sequence datasets, which are characterized by their high dimensionality and complexity. The choice of genomic data is motivated by its significance in understanding biological processes and its potential for impacting fields such as personalized medicine and evolutionary biology.

### Technical Framework

All experiments are conducted using Qiskit, an open-source quantum computing software development framework. Qiskit enables the design, simulation, and analysis of quantum algorithms, making it an ideal tool for this project. Qiskit Machine learning is used to develop machine learning models.
