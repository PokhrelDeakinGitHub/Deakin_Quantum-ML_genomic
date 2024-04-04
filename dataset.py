from pathlib import Path
import warnings
import os
import sys
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import tensorflow as tf
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import numpy as np
from genomic_benchmarks.loc2seq import download_dataset
from genomic_benchmarks.data_check import is_downloaded, info
from genomic_benchmarks.models.tf import vectorize_layer
from genomic_benchmarks.models.tf import get_basic_cnn_model_v0 as get_model

warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import time

warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")

#Load the dataset
DATASET = "demo_coding_vs_intergenomic_seqs"
if not is_downloaded(DATASET):
    download_dataset(DATASET)
    print("Dataset downloaded successfully")

# dataset_path = Path(os.path.join("datasets", DATASET))
# CLASSES = [x.stem for x in dataset_path.iterdir() if x.is_dir()]
# NUM_CLASSES = len(CLASSES)

dataset_path = Path.home() / '.genomic_benchmarks' / DATASET
CLASSES = [x.stem for x in (dataset_path/'train').iterdir() if x.is_dir()]
NUM_CLASSES = len(CLASSES)

warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")

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
VOCAB_SIZE_train = len(vectorize_layer.get_vocabulary())
vectorize_layer.get_vocabulary()
train_ds = train_ds.map(vectorize_text)

vectorize_layer.adapt(test_ds.map(lambda x, y: x))
VOCAB_SIZE_test = len(vectorize_layer.get_vocabulary())
vectorize_layer.get_vocabulary()
test_ds = test_ds.map(vectorize_text)





# Create the np_data_set list
np_train_data_set = []
for text_list, label_list in train_ds:
    for text, label in zip(text_list.numpy(), label_list.numpy()):
        sequence_dict = {"sequence": text.tolist(), "label": label.tolist()}
        np_train_data_set.append(sequence_dict)
# Convert the list of dictionaries to a NumPy array
np_train_data_set = np.array(np_train_data_set)

# Create the np_data_set list
np_test_data_set = []
for text_list, label_list in train_ds:
    for text, label in zip(text_list.numpy(), label_list.numpy()):
        sequence_dict = {"sequence": text.tolist(), "label": label.tolist()}
        np_test_data_set.append(sequence_dict)
# Convert the list of dictionaries to a NumPy array
np_test_data_set = np.array(np_test_data_set)

# print("\n\n")
# print("Print the first 5 examples in train dataset")
# for i, example in enumerate(np_train_data_set[:2]):
#     print(f"Example {i + 1} - Sequence: {example['sequence']}, Label: {example['label']}")

# print("\n\n")

# print("\n\n")
# print("Print the first 5 examples in test dataset")
# for i, example in enumerate(np_test_data_set[:2]):
#     print(f"Example {i + 1} - Sequence: {example['sequence']}, Label: {example['label']}")

print("\n\n")

train_sequences = [data_point["sequence"] for data_point in np_train_data_set]
train_labels = [data_point["label"] for data_point in np_train_data_set]
print(train_sequences[:2])
print(train_labels[:2])
print("\n\n")
print("\n\n")
test_sequences = [data_point["sequence"] for data_point in np_test_data_set]
test_labels = [data_point["label"] for data_point in np_test_data_set]
print("\n\n")
print("\n\n")
print(test_sequences[:2])
print(test_labels[:2])


print("\n\n")
print("\n\n")
train_sequences = np.array(train_sequences)
train_labels = np.array(train_labels)
test_sequences = np.array(test_sequences)
test_labels = np.array(test_labels)

print("after numpy array",train_sequences[:2])
print("after numpy array",train_labels[:2])
print("after numpy array",test_sequences[:2])
print("after numpy array",test_labels[:2])
