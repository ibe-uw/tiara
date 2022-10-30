from itertools import product
from collections import defaultdict
from numba.typed import Dict, List
from numba import types, njit

import numpy as np

MAX_K = 7

# A dictionary of dictionaries, specifying the position of a kmer in a vector.
kmer_to_pos = {
    k: Dict.empty(key_type=types.unicode_type, value_type=types.int64)
    for k in range(1, MAX_K + 1)
}
for k in range(1, MAX_K + 1):
    for i, kmer in enumerate(product("ACGT", repeat=k)):
        kmer_to_pos[k]["".join(kmer)] = i


def oligofreq(sequence: str, k: int) -> np.ndarray:
    """A function that calculates oligonucleotide frequency.

    In simpler terms, it calculates a bag-of-words representation of a sequence given k-mer length.

    Examples
    --------
    >>> oligofreq("AACT", 2)
    array([1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
      dtype=float32)

    Parameters
    ----------
    sequence: a nucleotide sequence to represent
    k: k-mer length

    Returns
    -------
    vector:
        a numpy array of length 4^k that has, at position j, the number of kmer in a sequence that is
        j-th in a lexicographic order
        (so, for example, vector[0] would represent, for k=5, a number of AAAAAs in a sequence).
    """
    vector = defaultdict(int)
    for pos in range(len(sequence) - k + 1):
        vector[sequence[pos : pos + k]] += 1
    return np.array(
        [vector["".join(kmer)] for kmer in product("ACGT", repeat=k)], dtype=np.float32
    )


def single_oligofreq(sequence, k):
    """Calculate a bag-of-words representation of a sequence, faster than oligofreq."""
    if k not in kmer_to_pos:
        raise ValueError("k-mer length not supported!")
    d = kmer_to_pos[k]
    seqs = List()
    seqs.append(sequence)
    return calc_array(seqs, d, k)


@njit
def calc_array(seqs, d, k):
    result = np.zeros((len(seqs), 4**k), dtype=np.float32)
    for i, seq in enumerate(seqs):
        for pos in range(len(seq) - k + 1):
            subseq = seq[pos : pos + k]
            if subseq in d:
                result[i, d[subseq]] += 1
    return result


def multiple_oligofreq(seqs, k):
    """Calculate bag-of-words representations of a list of sequences."""
    if k not in kmer_to_pos:
        raise ValueError("k-mer length not supported!")
    d = kmer_to_pos[k]
    return calc_array(seqs, d, k)
