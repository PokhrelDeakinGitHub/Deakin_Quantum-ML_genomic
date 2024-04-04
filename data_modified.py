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
import matplotlib.pyplot as plt
import time

warnings.filterwarnings("ignore")

# Load the dataset
DATASET = "demo_coding_vs_intergenomic_seqs"
if not is_downloaded(DATASET):
    download_dataset(DATASET)
    print("Dataset downloaded successfully")

dataset_path = Path.home() / '.genomic_benchmarks' / DATASET
CLASSES = [x.stem for x in (dataset_path/'train').iterdir() if x.is_dir()]
NUM_CLASSES = len(CLASSES)

train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    dataset_path/ 'train',
    batch_size=64,
    seed=1337,
    class_names=CLASSES,
)

CLASSES = [x.stem for x in (dataset_path/'test').iterdir() if x.is_dir()]
NUM_CLASSES = len(CLASSES)

test_ds = tf.keras.preprocessing.text_dataset_from_directory(
   dataset_path/ 'test',
    batch_size=64,
    seed=1337,
    class_names=CLASSES,
)

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text)-2, label

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