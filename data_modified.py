from pathlib import Path
import warnings
import os
import sys
import tensorflow as tf
import numpy as np
from genomic_benchmarks.loc2seq import download_dataset
from genomic_benchmarks.data_check import is_downloaded, info
from genomic_benchmarks.models.tf import vectorize_layer
from genomic_benchmarks.models.tf import get_basic_cnn_model_v0 as get_model
import time

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Function to download the dataset
def download_genomic_dataset(dataset_name):
    if not is_downloaded(dataset_name):
        download_dataset(dataset_name)
        print("Dataset downloaded successfully")

# Function to load the dataset
def load_dataset(dataset_path, batch_size, seed):
    CLASSES = [x.stem for x in dataset_path.iterdir() if x.is_dir()]
    NUM_CLASSES = len(CLASSES)

    ds = tf.keras.preprocessing.text_dataset_from_directory(
        dataset_path,
        batch_size=batch_size,
        seed=seed,
        class_names=CLASSES,
    )

    return ds

# Function to vectorize text and labels
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text)-2, label

# Function to preprocess and save the data
def preprocess_and_save_data(train_ds, test_ds):
    vectorize_layer.adapt(train_ds.map(lambda x, y: x))
    train_ds = train_ds.map(vectorize_text)

    vectorize_layer.adapt(test_ds.map(lambda x, y: x))
    test_ds = test_ds.map(vectorize_text)

    np_train_data_set = [{"sequence": text.tolist(), "label": label.tolist()} for text_list, label_list in train_ds for text, label in zip(text_list.numpy(), label_list.numpy())]
    
    np_test_data_set = [{"sequence": text.tolist(), "label": label.tolist()} for text_list, label_list in test_ds for text, label in zip(text_list.numpy(), label_list.numpy())]

    train_sequences = np.array([data_point["sequence"] for data_point in np_train_data_set])
    train_labels = np.array([data_point["label"] for data_point in np_train_data_set])
    test_sequences = np.array([data_point["sequence"] for data_point in np_test_data_set])
    test_labels = np.array([data_point["label"] for data_point in np_test_data_set])

    # Save the preprocessed data
    np.save('train_sequences.npy', train_sequences)
    np.save('train_labels.npy', train_labels)
    np.save('test_sequences.npy', test_sequences)
    np.save('test_labels.npy', test_labels)

# Main function
def main():
    # Load the dataset
    DATASET = "demo_coding_vs_intergenomic_seqs"
    dataset_path = Path.home() / '.genomic_benchmarks' / DATASET

    download_genomic_dataset(DATASET)

    train_ds = load_dataset(dataset_path / 'train', batch_size=64, seed=1337)
    test_ds = load_dataset(dataset_path / 'test', batch_size=64, seed=1337)

    preprocess_and_save_data(train_ds, test_ds)

if __name__ == "__main__":
    main()
