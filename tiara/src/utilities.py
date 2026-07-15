from typing import List, Dict
from collections import defaultdict
from dataclasses import dataclass
import pickle
import warnings
from contextlib import contextmanager
from typing import Optional
import time

import numpy as np
from skorch.dataset import Dataset

classes_list = [
    ["organelle", "bacteria", "archaea", "eukarya", "unknown"],
    ["plastid", "unknown", "mitochondrion"],
]


@dataclass
class SingleResult:
    """Class representing a single result.

    Parameters
    ----------
        cls - a list of classes (one entry per one stage of classification, e.g. ['organelle', 'plastid']
        desc - sequence description taken from fasta file
        seq - nucleotide sequence
        probs - a list of dictionaries mapping classes to probabilities assigned by the classifier
                (one dictionary per stage of classification)
        quality - fastq quality value of the sequence (optional)
    """

    cls: List[str]
    desc: str
    seq: str
    probs: List[Dict[str, float]]
    quality: Optional[str] = None

    def generate_line(self, prob=False):
        """Generates one line of the output summarizing the result."""
        res = "\t".join([self.desc, self.cls[0], self.cls[1]])
        if prob:
            res += "\t"
            res += "\t".join([f"{self.probs[0][cls]:.6f}" for cls in classes_list[0]])
            if not self.probs[1]:
                res += "\t" + "\t".join(["n/a", "n/a", "n/a"])
            else:
                res += "\t" + "\t".join(
                    [f"{self.probs[1][cls]:.6f}" for cls in classes_list[1]]
                )
        return res


class TransformedDataset(Dataset):
    def __init__(self, X, y, length, transformer, k: int):
        super().__init__(X, y, length)
        self.transformer = transformer
        self.k = k

    def __getitem__(self, i):
        seqs = self.X[i]
        labels = self.y[i]
        X = np.array([self.transformer.transform(seq) for seq in seqs]).reshape(
            (-1, 4**self.k)
        )
        return self.transform(X, labels)


def is_fastq(filename: str):
    return filename.lower().endswith((".fq.gz", ".fastq.gz", ".fq", ".fastq"))


def write_to_fasta(handle, seqs: List[SingleResult]):
    for record in seqs:
        handle.write(">" if record.quality is None else "@")
        handle.write(record.desc + "\n")
        handle.write(record.seq + "\n")
        if record.quality is not None:
            handle.write("+\n")
            handle.write(record.quality + "\n")


def sort_type(results):
    """Creates a dictionary with classes as keys and sequences belonging to classes as values."""
    sorted_seqs = defaultdict(list)
    for record in results:
        fst_iter, snd_iter = record.cls
        if fst_iter == "organelle":
            sorted_seqs[snd_iter].append(record)
        else:
            sorted_seqs[fst_iter].append(record)
    return sorted_seqs


def unpickle(file):
    """Unpickle something (in this case tf-id)."""
    with open(file, "rb") as source:
        return pickle.load(source)


def parse_params(fname: str) -> Dict[str, int]:
    """Take a comma separated csv file with no header, return dict with keys from col 1 and values from col 2.

    Example file:
       fragment_len,5000
       kmer,5
       hidden_1,128
       hidden_2,64
       dim_out,5

    Parameters
    ----------
        fname: a filename of a .csv file

    Returns
    -------
        ret: a dictionary of the form {param: value}, created from a .csv file
    """
    ret = {}
    with open(fname, "r") as source:
        for line in source:
            param, value = line.strip().split(",")
            ret[param.strip()] = value.strip()
    return ret


def count_sequences(fasta_file: str) -> int:
    """Count sequences in a fasta file."""
    n = 0
    with open(fasta_file, "r") as file:
        for line in file:
            if line.startswith(">"):
                n += 1
    return n


def chop(sequence: str, fragment_len: int) -> List[str]:
    """Chop a whole sequence into subsequences of length fragment_len.

    The subsequences are not-overlapping and their length is exactly fragment_len.
    so the ends of the sequences will usually be chopped off.

    Returns
    -------
        list: a list of subsequences
    """
    n_fragments = len(sequence) // fragment_len
    if n_fragments == 0:
        return [sequence]
    else:
        return [
            sequence[i * fragment_len : i * fragment_len + fragment_len]
            for i in range(n_fragments)
        ]


def merge_results(
    res1: List[SingleResult], res2: List[SingleResult]
) -> List[SingleResult]:
    """Merges results from first and second stage of classification."""
    result = []
    ids_in_second = set(record.desc for record in res2)
    for record in res1:
        if record.desc not in ids_in_second:
            result.append(record)
    for record in res2:
        result.append(record)
    return result


@contextmanager
def time_context_manager(label):
    """A context manager for timing.
    Taken from David Beazley's slides on generators ('Generators: The Final Frontier')
    """
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        if end - start > 60:
            if end - start > 60 * 60:
                print("{} took: {:f} hours".format(label, (end - start) / 60 * 60))
            else:
                print("{} took: {:f} minutes".format(label, (end - start) / 60))
        else:
            print("{} took: {:f} seconds".format(label, (end - start)))
