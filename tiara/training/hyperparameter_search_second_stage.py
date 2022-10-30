from itertools import product
import time
import sys
import os
import json

import numpy as np
import pandas as pd
import pkg_resources
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.callbacks import EpochScoring, ProgressBar
import torch
from torch import nn
from skorch.dataset import Dataset
from Bio.SeqIO.FastaIO import SimpleFastaParser
from numba import njit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from tiara.src.transformations import TfidfWeighter
from tiara.src.transformations import Transformer

torch.set_num_threads(int(sys.argv[4]))

INPUT_DIR = sys.argv[1]
OUTPUT_FILE = sys.argv[2]
k = int(sys.argv[3])


@njit
def add_ones_matrix(mat, row, positions):
    for j in range(len(positions)):
        mat[row, positions[j]] += 1


def get_tfidf_repr(seqs, k, idf_vec):
    length = len(seqs)
    result = np.zeros((length, 4**k), dtype=np.float32)
    for i, seq in enumerate(seqs):
        arr = []
        for pos in range(len(seq) - k + 1):
            try:
                arr.append(kmer_to_pos_mappings[k][seq[pos : pos + k]])
            except KeyError:
                continue
        positions = np.array(arr)
        add_ones_matrix(result, i, positions)
    return result * idf_vec


architectures = []
for hid in [32, 64, 128, 256]:
    for learning_rate in [0.1, 0.01, 0.001, 0.0001]:
        for dropout in [0.2, 0.5]:
            architectures.append(
                dict(hid1=hid, learning_rate=learning_rate, dropout=dropout)
            )
            architectures.append(
                dict(hid1=hid, hid2=hid, learning_rate=learning_rate, dropout=dropout)
            )
for learning_rate in [0.1, 0.01, 0.001, 0.0001]:
    for dropout in [0.2, 0.5]:
        architectures.append(
            dict(hid1=64, hid2=32, learning_rate=learning_rate, dropout=dropout)
        )
        architectures.append(
            dict(hid1=128, hid2=64, learning_rate=learning_rate, dropout=dropout)
        )
        architectures.append(
            dict(hid1=256, hid2=128, learning_rate=learning_rate, dropout=dropout)
        )


tfidf_filepath = pkg_resources.resource_filename(
    __name__, f"../models/tfidf-models/k{k}-second-stage"
)

tfidf = TfidfWeighter.load_params(tfidf_filepath)
print("Started importing data")
with open(os.path.join(INPUT_DIR, "train/mitochondria.fasta"), "r") as handle:
    train_mitochondria = [seq for _, seq in SimpleFastaParser(handle)]
with open(os.path.join(INPUT_DIR, "train/plastids.fasta"), "r") as handle:
    train_plastids = [seq for _, seq in SimpleFastaParser(handle)]
with open(os.path.join(INPUT_DIR, "validation/mitochondria.fasta"), "r") as handle:
    val_mitochondria = [seq for _, seq in SimpleFastaParser(handle)]
with open(os.path.join(INPUT_DIR, "validation/plastids.fasta"), "r") as handle:
    val_plastids = [seq for _, seq in SimpleFastaParser(handle)]


print("Done")
transformer = Transformer(tfidf, fragment_len=5000, k=k)

kmer_to_pos_mappings = {
    k: {"".join(kmer): i for i, kmer in enumerate(product("ACGT", repeat=k))}
    for k in range(1, 10)
}


transformer.model.idfs = transformer.model.idfs.astype(np.float32)


print("Started computing representations for train data")
start_time = time.time()
train_X = get_tfidf_repr(train_plastids + train_mitochondria, k, transformer.model.idfs)
print(f"Computing representations took: {time.time() - start_time}s.")

print("Started computing representations for validation data")
start_time = time.time()
val_X = get_tfidf_repr(val_plastids + val_mitochondria, k, transformer.model.idfs)
print(f"Computing representations took: {time.time() - start_time}s.")

train_y = np.array([0] * len(train_plastids) + [2] * len(train_mitochondria))
val_y = np.array([0] * len(val_plastids) + [2] * len(val_mitochondria))

valid_dataset = Dataset(val_X, val_y)

train_X = train_X / np.linalg.norm(train_X, axis=1).reshape((-1, 1))
val_X = val_X / np.linalg.norm(val_X, axis=1).reshape((-1, 1))

dim_in = 4**k
dim_out = 3

with open(sys.argv[1], "a") as handle:
    handle.write("0: plastids, 1: unknown, 2: mitochondria\n")

for i, architecture in enumerate(architectures):
    hid1 = architecture["hid1"]
    try:
        hid2 = architecture["hid2"]
    except KeyError:
        hid2 = None
    lr = architecture["learning_rate"]
    drop = architecture["dropout"]
    print(f"Starting testing architecture number {i+1}/{len(architectures)}")

    def get_metrics(y_test, y_predicted):
        accuracy = accuracy_score(y_test, y_predicted)
        precision = precision_score(y_test, y_predicted, average=None)
        recall = recall_score(y_test, y_predicted, average=None)
        f1 = f1_score(y_test, y_predicted, average=None)
        return accuracy, precision, recall, f1

    def mean_f1(net, ds, y=None):
        y_true = val_y
        y_pred = net.predict(val_X)
        res = {
            key: val
            for key, val in zip(
                ["accuracy", "precision", "recall", "f1"], get_metrics(y_true, y_pred)
            )
        }
        net.history.record("all_metrics", {key: str(val) for key, val in res.items()})
        return np.mean(res["f1"])

    if hid2:
        print(f"Parameters: hid1: {hid1}, hid2: {hid2}, lr: {lr}, drop: {drop}")

        class MyNNet_2(nn.Sequential):
            def __init__(self, dim_in, hid1, hid2, dim_out):
                super().__init__(
                    nn.Linear(dim_in, hid1),
                    nn.Dropout(drop),
                    nn.ReLU(inplace=True),
                    nn.Linear(hid1, hid2),
                    nn.Dropout(drop),
                    nn.ReLU(inplace=True),
                    nn.Linear(hid2, dim_out),
                    nn.Softmax(1),
                )

        net = NeuralNetClassifier(
            MyNNet_2(dim_in, hid1, hid2, dim_out),
            max_epochs=50,
            lr=lr,
            train_split=predefined_split(valid_dataset),
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
            optimizer=torch.optim.Adam,
            verbose=10,
            callbacks=[EpochScoring(mean_f1, use_caching=False), ProgressBar()],
        )
    else:
        print(f"Parameters: hid1: {hid1}, lr: {lr}, drop: {drop}")

        class MyNNet_2(nn.Sequential):
            def __init__(self, dim_in, hid1, dim_out):
                super().__init__(
                    nn.Linear(dim_in, hid1),
                    nn.Dropout(drop),
                    nn.ReLU(inplace=True),
                    nn.Linear(hid1, dim_out),
                    nn.Softmax(1),
                )

        net = NeuralNetClassifier(
            MyNNet_2(dim_in, hid1, dim_out),
            max_epochs=50,
            lr=lr,
            train_split=predefined_split(valid_dataset),
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
            optimizer=torch.optim.Adam,
            verbose=10,
            callbacks=[EpochScoring(mean_f1, use_caching=False), ProgressBar()],
        )

    net.fit(train_X, train_y)

    y_pred = net.predict(val_X)

    print("Confusion matrix\n")
    print(
        pd.crosstab(
            pd.Series(val_y, name="Actual"), pd.Series(y_pred, name="Predicted")
        )
    )

    accuracy, precision, recall, f1 = get_metrics(val_y, y_pred)

    print(accuracy)
    print(precision)
    print(recall)
    print(f1)

    with open(OUTPUT_FILE, "a") as handle:

        handle.write(
            f"Starting testing architecture number {i+1}/{len(architectures)}\n"
        )
        if hid2:
            handle.write(
                f"Parameters: hid1: {hid1}, hid2: {hid2}, lr: {lr}, drop: {drop}\n"
            )
        else:
            handle.write(f"Parameters: hid1: {hid1}, lr: {lr}, drop: {drop}\n")
        handle.write(
            f"{pd.crosstab(pd.Series(val_y, name='Actual'), pd.Series(y_pred, name='Predicted'))}\n"
        )
        for metric_name, metric in zip(
            ["accuracy", "precision", "recall", "f1"], [accuracy, precision, recall, f1]
        ):
            handle.write(f"{metric_name}: {metric}\n")
        handle.write("\n")

    directory, filename = os.path.split(OUTPUT_FILE)
    with open(os.path.join(directory, "histories_" + filename), "a") as handle:
        if hid2:
            handle.write(
                f"Parameters: hid1: {hid1}, hid2: {hid2}, lr: {lr}, drop: {drop}\n"
            )
        else:
            handle.write(f"Parameters: hid1: {hid1}, lr: {lr}, drop: {drop}\n")
        json.dump(net.history.to_list(), handle)
        handle.write("\n")
