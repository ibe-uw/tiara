import os
import sys

from tiara.src.transformations import TfidfWeighter

# replace with your real data
DATA_PREFIX = sys.argv[1]
OUTPUT_PREF = sys.argv[2]
os.makedirs(OUTPUT_PREF)


data_first_step = [
    os.path.join(DATA_PREFIX, f"{file}.fasta")
    for file in ["mitochondria", "plastids", "bacteria", "eukarya", "archaea"]
]
data_second_step = [
    os.path.join(DATA_PREFIX, f"{file}.fasta") for file in ["mitochondria", "plastids"]
]

for kmer in [4, 5, 6]:
    tfidf = TfidfWeighter(k=kmer, fragment_len=5000, verbose=True)
    tfidf.fit(data_first_step)
    tfidf.save_params(params_fname=os.path.join(OUTPUT_PREF, f"k{kmer}-first-stage"))

for kmer in [4, 5, 6, 7]:
    tfidf = TfidfWeighter(k=kmer, fragment_len=5000, verbose=True)
    tfidf.fit(data_second_step)
    tfidf.save_params(params_fname=os.path.join(OUTPUT_PREF, f"k{kmer}-second-stage"))
