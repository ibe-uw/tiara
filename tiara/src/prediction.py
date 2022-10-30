from typing import Tuple, Dict, List
import warnings

from skorch import NeuralNetClassifier
import numpy as np

from tiara.src.utilities import chop, SingleResult


classes_mapping = {
    "organelle": 0,
    "bacteria": 1,
    "mitochondrion": 2,
    "archaea": 3,
    "eukarya": 4,
    "plastid": 0,
    "unknown": None,
}

id_to_class = {
    0: {0: "organelle", 1: "bacteria", 3: "archaea", 4: "eukarya", 2: "unknown"},
    1: {0: "plastid", 1: "unknown", 2: "mitochondrion"},
}


def predict_with_threshold(
    probs: np.ndarray, record: Tuple[str, str], prob_cutoff: float, layer: int
) -> SingleResult:
    """Predict the class.

    Parameters
    ----------
        probs: an array of probabilities of belonging to each class (length 5 for layer 1, 3 for layer 2)
        record: a tuple of strings (sequence description, sequence)
        prob_cutoff: a threshold for classifying to a class
        layer: layer indice (0 or 1)
    """
    desc, seq = record
    counts = {id_to_class[layer][i]: value for i, value in enumerate(probs)}
    chosen_class, prob_value = max(counts.items(), key=lambda x: x[1])
    if prob_value > prob_cutoff:
        if layer == 0:
            return SingleResult(
                cls=[chosen_class, "n/a"], desc=desc, seq=seq, probs=[counts, {}]
            )
        else:
            return SingleResult(
                cls=["organelle", chosen_class], desc=desc, seq=seq, probs=[{}, counts]
            )
    elif layer == 0 and counts["archaea"] + counts["bacteria"] > prob_cutoff:
        return SingleResult(
            cls=["prokarya", "n/a"], desc=desc, seq=seq, probs=[counts, {}]
        )
    else:
        if layer == 0:
            return SingleResult(
                cls=["unknown", "n/a"], desc=desc, seq=seq, probs=[counts, {}]
            )
        else:
            return SingleResult(
                cls=["organelle", "unknown"], desc=desc, seq=seq, probs=[{}, counts]
            )


class Prediction:
    """Performs a prediction based on supplied single record.

    Methods:
        make_prediction: a method that takes a single record and returns a prediction
    """

    def __init__(
        self,
        # min_percent: float,
        fragment_len: int,
        prob_cutoff: float,
        layer: int,
        nnet: NeuralNetClassifier,
        k: int,
        tnf,
        transformer,
    ):
        """Init method.

        Parameters
        ----------
            fragment_len: a length of individual fragment of a whole sequence
            prob_cutoff: probability at which a sequence is classified to a class
            layer: current phase of classification
            nnet: a skorch neural net object
            kmer: kmer length
            tfidf: tf-idf model
        """
        self.fragment_len = fragment_len
        self.prob_cutoff = prob_cutoff
        self.layer = layer
        self.nnet = nnet
        self.k = k
        self.tfidf = tnf
        self.transformer = transformer

    def make_prediction(
        self, single_record: Tuple[str, str, np.ndarray]
    ) -> SingleResult:
        """Make a prediction on a single sequence.

        The decision rule works as follows:
            1. Perform a prediction on a list of vectors representing fragments of sequences
            2. The resulting matrix with shape (number of fragments, number of classes) represents
            at position (i, j) a probability that fragment i belongs to class j.
            3. The mean of the matrix is taken, along the axis 0, which results in a vector
            of length equal to number of classes.
            4. The maximum of the vector is picked. If it doesn't exceed self.prob_cutoff, then
            another possibility is considered: that individually bacteria and archea classes do not
            exceed self.prob_cutoff, but together they do. If that's the case, the record is classified to
            a general "prokarya" class. Else, the sequence is classified as "unknown".

        Parameters
        ----------
            single_record:
                input sequence with its id (a tuple (sequence_id, sequence))

        Returns
        -------
            prediction:
                A SingleResult class instance.
        """
        desc, seq, bow = single_record
        # data = chop(seq, fragment_len=self.fragment_len)
        nnet_predictions = self.nnet.predict_proba(bow)
        mean_predictions = np.mean(nnet_predictions, axis=0)
        return predict_with_threshold(
            mean_predictions, (desc, seq), self.prob_cutoff, self.layer
        )
