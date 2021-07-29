# Usage

-----

If you installed **tiara** using our [instructions](detailed-installation.md), you have to make sure
you are in the environment you created. So before running **tiara** you have to run `conda activate tiara-env`,
assuming that you've named the environment `tiara-env`.

### Basic usage:
```bash
tiara -i sample_input.fasta -o out.txt
```

The sequences in the fasta file should be at least 3000 bases long (default value). We do not recommend classifying 
sequences shorter than 1000 base pairs.

It creates two files: 
 - out.txt, a tab-separated file with header `sequence id, first stage classification result, second stage classification result`.
 - log_out.txt, containing model parameters and classification summary.

### Advanced:

```bash
tiara -i sample_input.fasta -o out.txt --tf mit pla pro -t 4 -p 0.65 0.60 --probabilities
```

In addition to creating the files above, it creates, in the folder where `tiara` is run,
three files containing sequences from `sample_input.fasta` classified as 
mitochondria, plastid and prokarya (`--tf mit pla pro` option).

The number of threads is set to 4 (`-t 4`) and probability cutoffs 
in the first and second stage of classification are set to 0.65 and 0.6, respectively.

The probabilities of belonging to individual classes are also 
written to `out.txt`, thanks to `--probabilities` option.

### Program options (any order is fine)

- `-i input`, `--input input` A path to an input fasta file.
- `-o output`, `--output output` A path to output file. If not provided, the result is printed to stdout.
- `-m MIN_LEN`, `--min_len MIN_LEN` Minimum length of a sequence. Default: 3000.
- `-p cutoff [cutoff ...]`, `--prob_cutoff cutoff [cutoff ...]` Probability threshold needed for classification to a class.
    If two floats are provided, the first is used in a first stage, the second in the second stage
    Default: `[0.65, 0.65]`.
- `--to_fasta class [class ...]`, `--tf class [class ...]` Write sequences to fasta files specified in the arguments to this option.
    The arguments are: `mit` - mitochondria, `pla` - plastid, `bac` - bacteria,
    `arc` - archea, `euk` - eukarya, `unk` - unknown, `pro` - prokarya,
    `all` - all classes present in input fasta (to separate fasta files).
- `-t THREADS`, `--threads THREADS` Number of threads used.
- `--probabilities`, `--pr` Whether to write probabilities of individual classes for each sequence.
- `-v`, `--verbose` Whether to display some additional messages and progress bar during classification.
- `--gz`, `--gzip` gzip all program outputs (adds `.gz` extension to `-o`).
##### Advanced options
- `--first_stage_kmer`, `--k1` k-mer length used in the first stage of classification.
 Default: 6. Available options: `4, 5, 6`.
- `--second_stage_kmer`, `--k2` k-mer length used in the second stage of classification.
Default: 7. Available options: `4, 5, 6, 7`.
  

### Running training scripts

#### TF-IDF

You can train your own TF-IDF models using a following command (run from the repo):

```bash
python -m tiara.training.train_tfidf <direcory with input fasta files> <output directory>
```

Input fasta files have to include `archaea.fasta`, `eukarya.fasta`, 
`mitochondria.fasta`, `plastids.fasta` and `bacteria.fasta`. 
Editing the script to suit other needs should be easy.

#### Training neural networks

Run 
```bash
python -m tiara.training.train_models <input dir> <output dir> <n_cores>
```
This will produce models for best parameters for each k-mer length (`4-6` for the first stage, `4-7` for the second).
The `<input dir>` has to have the same structure as the output dir.

#### Hyperparameter search 

The input fasta directory has to have two directories: `train` and `validation`. 
Each one has to have either all types of inputs described above or only `mitochondria.fasta` and `plastids.fasta`,
depending on the stage of classification.

Hyperparameter search using the default TF-IDF models can be done calling:
```bash
python -m tiara.training.hyperparameter_search_first_stage <input dir> <output filename> <kmer length> <n_cores>
```
Similarly for `tiara.training.hyperparameter_search_second_stage`.

The output file includes a confusion matrix and some statistics
like accuracy, F1 etc calculated at the end of the default 50-epoch training.
There is also additional file produced (with `histories_` prefix added), with statistics calculated after each epoch.


### Using **tiara** as a package

Although most of the code is specific to our tool, 
the `TfidfWeighter` class could be useful for any researcher wanting to represent 
nucleotide sequences with oligonucleotide frequency (basically bag-of-words), 
weighted with [tf-idf weighting](https://en.wikipedia.org/wiki/Tf–idf).
The algorithm is explained in the docstrings. We provide several models for different k-mer lengths 
in the `tiara/models/tfidf-models` folder.

The basic usage would be:

```python
>>> from tiara import TfidfWeighter

# to read some more documentation
>>> help(TfidfWeighter)

# k means k-mer length
# sequences longer than fragment_len will be splitted to fragments
# so they will be treated as separate words in tf-idf
>>> tfidf = TfidfWeighter(k=5, fragment_len=5000)
>>> tfidf.fit(["list", "of", "fasta", "files"]) # or filename of a single fasta file
>>> transformed_sequences = tfidf.transform(["ACCGTTTGCAC", "AACGCGACGTGCGAGTTT"]) # or a single nucleotide sequence

# saving the model
# it creates two files: params.txt with hyperparameters of the model (k, fragment_len etc)
# and model.npy, a numpy array containing idf vector
# in a directory specified in the argument
# if no directory is specified it will generate the name automatically, 
# based on data names used to train the model and other parameters
>>> tfidf.save_params("model_directory") 

# loading the model
>>> tfidf = TfidfWeighter.load_params("model_directory")
```

There are also a couple of functions which calculate 
the oligonucleotide frequency of a sequence, or a set of sequences.

```python
>>> from tiara import oligofreq, single_oligofreq, multiple_oligofreq

>>> sample_sequence = "AGCTGCGCGACGCGAGCGTGCGCT"
# the oligofreq function has a simple implementation, works on any kmer, but is slower than the other ones
>>> oligofreq(sample_sequence, 2) 
array([0., 1., 2., 0., 0., 0., 6., 2., 2., 7., 0., 1., 0., 0., 2., 0.],
      dtype=float32)
# the functions single_oligofreq and multiple_oligofreq are faster than oligofreq
# but they utilize a "hardcoded" dictionary that tells the position of each kmer in an array
# so they work on kmer lengths from 1 to 7
# they utilize numba under the hood
>>> single_oligofreq(sample_sequence, 2)
array([0., 1., 2., 0., 0., 0., 6., 2., 2., 7., 0., 1., 0., 0., 2., 0.],
      dtype=float32)
>>> multiple_oligofreq([sample_sequence, sample_sequence], 2)
array([[0., 1., 2., 0., 0., 0., 6., 2., 2., 7., 0., 1., 0., 0., 2., 0.],
       [0., 1., 2., 0., 0., 0., 6., 2., 2., 7., 0., 1., 0., 0., 2., 0.]],
      dtype=float32)
```

[Back to README](README.md)
