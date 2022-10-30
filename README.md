# [Tiara](https://ibe-uw.github.io/tiara/)

Deep-learning-based approach for identification of eukaryotic sequences in the metagenomic data powered by [PyTorch](https://pytorch.org).  

The sequences are classified in two stages:

- In the first stage, the sequences are classified to classes: 
      archaea, bacteria, prokarya, eukarya, organelle and unknown.
- In the second stage, the sequences labeled as organelle in the first stage 
      are classified to either mitochondria, plastid or unknown.

For more information, please refer to our paper:
[*Tiara: Deep learning-based classification system for eukaryotic sequences*](https://doi.org/10.1093/bioinformatics/btab672).

[Supplementary data](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btab672/6375939#supplementary-data)

[Supplementary sequences](data/Supplementary_sequences)

## Requirements

- `Python >= 3.7`
- `numpy, biopython, torch, skorch, tqdm, joblib, numba`

## Installation

More detailed installation instructions can be found [here](docs/detailed-installation.md).

#### Using `pip`

Run `pip install tiara`, preferably in a fresh environment.

#### Using setup.py

##### Latest stable release

- Download latest release from https://github.com/ibe-uw/tiara/releases.
- Unzip/untar the archive.
- Go to the directory.
- Run `python setup.py install`.

##### Latest developer version

```bash
git clone https://github.com/ibe-uw/tiara.git
cd tiara
python setup.py install
```

#### Testing the installation

After the installation, run `tiara-test` to see if the installation was successful.

## Usage

#### Basic usage:
```bash
tiara -i sample_input.fasta -o out.txt
```

The sequences in the fasta file should be at least 3000 bases long (default value). We do not recommend classify sequences that are shorter than 1000 base pairs.

It creates two files: 

 - out.txt, a tab-separated file with header `sequence id, first stage classification result, second stage classification result`.
 - log_out.txt, containing model parameters and classification summary.

#### Advanced:

```bash
tiara -i sample_input.fasta -o out.txt --tf mit pla pro -t 4 -p 0.65 0.60 --probabilities
```

In addition to creating the files above, it creates, in the folder where `tiara` is run,
three files containing sequences from `sample_input.fasta` classified as 
mitochondria, plastid and prokarya (`--tf mit pla pro` option).

The number of threads is set to 4 (`-t 4`) and probability cutoffs 
in the first and second stage of classification are set to 0.65 and 0.6, respectively.

The probabilities of belonging to individual classes are also written to 
`out.txt`, thanks to `--probabilities` option.

For more usage examples, go [here](docs/usage.md).

## Citation 

Michał Karlicki, Stanisław Antonowicz, Anna Karnkowska, Tiara: deep learning-based classification system for eukaryotic sequences, Bioinformatics, Volume 38, Issue 2, 15 January 2022, Pages 344–350, https://doi.org/10.1093/bioinformatics/btab672

## License

Tiara is released under an open-source MIT license 

### Version history:

- `1.0.3` – added `pyproject.toml`, updated dependencies to `python<3.10`
 – unfortunately `tiara` doesn't work right now with 
 `python` newer than `3.9` due to `torch 1.7.0` compatibility issues. 
  Added option to use gzipped fasta file as input (automatically identified by `.gz` suffix).
- `1.0.2` – added `Python 3.9` compatibility, added an option to gzip the results. 
  Added this README section.
- `1.0.0`, `1.0.1` – initial releases.













