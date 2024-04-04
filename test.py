import warnings
import os
import sys
import tensorflow as tf
import numpy as np
from pathlib import Path
from genomic_benchmarks.loc2seq import download_dataset
from genomic_benchmarks.data_check import is_downloaded, info
from genomic_benchmarks.models.tf import vectorize_layer
from genomic_benchmarks.models.tf import get_basic_cnn_model_v0 as get_model
import time

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")

def download_genomic_dataset(dataset):
    if not is_downloaded(dataset):
        download_dataset(dataset)
        print("Dataset downloaded successfully")

def load_genomic_dataset(dataset_path):
    classes = [x.stem for x in dataset_path.iterdir() if x.is_dir()]
    num_classes = len(classes)

    ds = tf.keras.preprocessing.text_dataset_from_directory(
        dataset_path,
        batch_size=64,
        seed=1337,
        class_names=classes,
    )

    return ds, classes, num_classes

def vectorize_text(text, label, vectorize_layer):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text)-2, label

def print_data(data, label, num_datapoints):
    print(f"\nPrinting {num_datapoints} examples:")
    for i, example in enumerate(zip(data[:num_datapoints], label[:num_datapoints])):
        print(f"Example {i + 1} - Sequence: {example[0]}, Label: {example[1]}")

def main():
    # Load the dataset
    dataset = "demo_coding_vs_intergenomic_seqs"
    dataset_path = Path.home() / '.genomic_benchmarks' / dataset

    download_genomic_dataset(dataset)
    train_ds, train_classes, train_num_classes = load_genomic_dataset(dataset_path / 'train')
    test_ds, test_classes, test_num_classes = load_genomic_dataset(dataset_path / 'test')

    vectorize_layer.adapt(train_ds.map(lambda x, y: x))
    VOCAB_SIZE_train = len(vectorize_layer.get_vocabulary())
    train_ds = train_ds.map(lambda x, y: vectorize_text(x, y, vectorize_layer))

    vectorize_layer.adapt(test_ds.map(lambda x, y: x))
    VOCAB_SIZE_test = len(vectorize_layer.get_vocabulary())
    test_ds = test_ds.map(lambda x, y: vectorize_text(x, y, vectorize_layer))

    # Create the np_train_data_set list
    np_train_data_set = []
    for text_list, label_list in train_ds:
        for text, label in zip(text_list.numpy(), label_list.numpy()):
            sequence_dict = {"sequence": text.tolist(), "label": label.tolist()}
            np_train_data_set.append(sequence_dict)
    np_train_data_set = np.array(np_train_data_set)

    # Create the np_test_data_set list
    np_test_data_set = []
    for text_list, label_list in test_ds:
        for text, label in zip(text_list.numpy(), label_list.numpy()):
            sequence_dict = {"sequence": text.tolist(), "label": label.tolist()}
            np_test_data_set.append(sequence_dict)
    np_test_data_set = np.array(np_test_data_set)

    print_data(np_train_data_set["sequence"], np_train_data_set["label"], 5)
    print_data(np_test_data_set["sequence"], np_test_data_set["label"], 5)

if __name__ == "__main__":
    main()
