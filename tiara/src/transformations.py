from collections import defaultdict
from typing import List, Union, Iterable
from itertools import product
import os
import tqdm

import numpy as np
from numba.typed import List as NumbaList
from Bio.SeqIO.FastaIO import SimpleFastaParser

from tiara.src.utilities import chop, count_sequences
from tiara.src.bow import single_oligofreq, multiple_oligofreq


class TfidfWeighter:
    """A class performing bag-of-words and tf-idf weighting of nucleotide sequences."""

    def __init__(self, k: int, fragment_len: int, verbose=False, smooth=True) -> None:
        """A tf-idf weighter class.

        Trains an idf vector, transforms a sequence or a list of sequences based on learned vector.

        Methods
        -------
            fit:
                Fits a model based on a fasta file or a list of fasta files provided.
            load_params:
                Reads a .npy file containing an idf vector.
            save_params:
                Saves idf vector to a file.
            transform:
                Transform provided sequence(s) in tf-idf representations.

        Parameters
        ----------
            k: kmer length
            fragment_len: subsequence fragment length (e.g. 5000)
            smooth: whether to add +1 in numerator and denominator in idf calculation or not
        """
        self.idfs = None
        self.k = k
        self.smooth = smooth
        self.N = 0
        self.data_names = None
        self.fragment_len = fragment_len
        self.verbose = verbose

    def __len__(self):
        return self.N

    def fit(self, data: Union[List[str], str]) -> None:
        """Fit a Tfidf model based on fasta data provided. Modifies self.idfs attribute.

        Calculates an idf vector of kmers in a sequences from data.
        idf(kmer, data) = ln(number_of_sequences/number of sequences in which the kmer appears) + 1)
        if self.smooth == True it adds 1 to numerator and denominator.

        Parameters
        ----------
            data: either a list of fasta filenames or a single filename
        """
        kmer_counts = defaultdict(int)

        if isinstance(data, str):
            fnames = [data]
        else:
            fnames = data
        self.data_names = [fname.split(".")[0] for fname in fnames]
        if self.verbose:
            print(
                f"Training tf-idf model on {self.data_names} files with kmer length {self.k}."
            )
        self.N = 0
        for i, fname in enumerate(fnames):
            if self.verbose:
                print(f"Processing file {i+1} ({fname})")
            with open(fname, "r") as handle:
                if self.verbose:
                    sequences = tqdm.tqdm(
                        enumerate(SimpleFastaParser(handle)),
                        total=count_sequences(fname),
                    )
                else:
                    sequences = enumerate(SimpleFastaParser(handle))
                for j, (_, sequence) in sequences:
                    if len(sequence) > self.fragment_len:
                        seqs = chop(sequence, self.fragment_len)
                    else:
                        seqs = [sequence]
                    for seq in seqs:
                        self.N += 1
                        kmer_encountered = defaultdict(bool)
                        for pos in range(len(seq) - self.k + 1):
                            kmer = seq[pos : pos + self.k]
                            if not kmer_encountered:
                                kmer_counts[kmer] += 1
                                kmer_encountered[kmer] = True
        self.idfs = np.array(
            [kmer_counts["".join(kmer)] for kmer in product("ACGT", repeat=self.k)],
            dtype=np.float32,
        )
        if self.smooth:
            self.idfs = np.log((self.N + 1) / (self.idfs + 1)) + 1
        else:
            self.idfs = np.log(self.N / self.idfs) + 1
        self.idfs = self.idfs.astype(np.float32)

    @classmethod
    def load_params(cls, folder_name: str) -> "TfidfWeighter":
        """Create a new instance of a class from saved parameters.

        The method assumes that in the argument folder there exist a params.txt file
        and a model.npy file, the former containing parameters such as kmer length etc,
        the latter being a saved numpy array containing idf weights.
        Example params.txt file:
            k:7
            fragment_len:5000
            verbose:True
            smooth:True
            N:1005231
            data_names:<comma separated paths to training files>

        Parameters
        ----------
            folder_name: name of the folder containing parameter files.

        Returns
        -------
            tfidf: an instance of TfidfWeighter class.
        """
        params_dict = {}
        with open(os.path.join(folder_name, "params.txt"), "r") as source:
            for line in source:
                key, val = line.strip().split(":")
                if key == "data_names":
                    val = val.split(",")
                params_dict[key] = val
        tfidf = TfidfWeighter(
            k=int(params_dict["k"]),
            fragment_len=int(params_dict["fragment_len"]),
            verbose=True if params_dict["verbose"] == "True" else False,
            smooth=True if params_dict["smooth"] == "True" else False,
        )
        tfidf.N = params_dict["N"]
        tfidf.data_names = params_dict["data_names"]
        tfidf.idfs = np.load(os.path.join(folder_name, "model.npy")).astype(np.float32)
        return tfidf

    def save_params(self, params_fname: str = None) -> None:
        """Saves parameters to files params.txt and model.npy in a folder params_fname.

        The folder path is either user-specified or generated automatically based on attributes.

        Parameters
        ----------
            params_fname (optional): folder name in which parameters file should be saved
        """
        if not params_fname:
            fname = f"tfidf_model_{'_'.join(os.path.split(fname)[1][:10] for fname in self.data_names)}_kmer={self.k}"
        else:
            fname = params_fname
        if not os.path.exists(fname):
            os.mkdir(fname)
        np.save(os.path.join(fname, "model"), self.idfs)
        # k: int, fragment_len: int, verbose=False, smooth=True
        with open(os.path.join(fname, "params.txt"), "w") as handle:
            handle.write(f"k:{self.k}\n")
            handle.write(f"fragment_len:{self.fragment_len}\n")
            handle.write(f"verbose:{self.verbose}\n")
            handle.write(f"smooth:{self.smooth}\n")
            handle.write(f"N:{self.N}\n")
            handle.write(f"data_names:{','.join(self.data_names)}")
        print(f"Model parameters saved to: {fname} folder")

    def transform(self, data: Union[str, Iterable[str]]) -> np.ndarray:
        """Transform nucleotide sequences to tf-idf weighted bow representations

        Takes either a single sequence (string) or an iterable of them.

        Parameters
        ----------
        data: a sequence or list of sequences (strings)

        Returns
        -------
        array: an array containing (possibly one) tf-idf weighted representation(s).
        """
        seqs = NumbaList()
        if isinstance(data, str):
            seqs.append(data)
        elif isinstance(data, Iterable):
            for seq in data:
                seqs.append(seq)
        else:
            raise TypeError(
                f"Expected either a str or iterable of str, got {type(data)} instead."
            )
        result = multiple_oligofreq(seqs, self.k) * self.idfs
        result = result / np.linalg.norm(result, axis=1).reshape((-1, 1))
        result = result.reshape(len(data), -1)
        return result


class Transformer:
    """A class that stores information about kmer length, fragment length and weighting model.

    A unified interface to a model doing the representation.
    The model's transform method should take a single nucleotide sequence or an iterable of them as an argument.

    Methods
    -------
    transform: transforms a nucleotide sequence into a representations using a model."""

    def __init__(self, model, fragment_len: int, k: int):
        self.model = model
        self.fragment_len = fragment_len
        self.k = k

    def transform(self, record: Union[str, Iterable[str]]):
        """Transform a nucleotide sequence or an iterable of them into a representation.

        All logic is done by the model.

        Parameters
        ----------
        record: A nucleotide sequence as a Python string or an iterable of them.

        Returns
        -------
        array: a self.model's representation of a record.
        """
        return self.model.transform(record)
