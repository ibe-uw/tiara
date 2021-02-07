# Tiara

Deep-learning-based approach for identification of eukaryotic sequences in the metagenomic data powered by PyTorch.  

The sequences are classified in two stages:

- In the first stage, the sequences are classified to classes 
      archaea, bacteria, prokarya, eukarya, organelle and unknown.
- In the second stage, the sequences labeled as organelle in the first stage 
      are classified to either mitochondria, plastid or unknown.

## Installation

### Requirements

- `Python >= 3.7`
- `numpy, biopython, torch, skorch, tqdm, joblib, numba`

### Quick installation

Detailed instructions can be found [here](detailed-installation.md).

#### Using setup.py

```bash
git clone https://github.com/ibe-uw/tiara.git
cd tiara
python setup.py install
```

This will install **tiara** in your Python environment.

## Usage

Sample usage can be found [here](usage.md).

### Sample pipelines

Here we describe some pipelines to tackle metagenomic data that utilize **tiara**. 
- [Eukaryotic MAGs recovery](eukaryotic_pipeline.md)
- [Organellar fraction recovery](organellar_pipeline.md)
- [Extracting prokaryotic fraction](prokaryotic_pipeline.md)

## Citing Tiara



## License

Tiara is released under an open-source MIT license 

### Name

In the Polish translation of the Harry Potter book series, the Sorting Hat
(which assigned wizards to different houses) was called *Tiara Przydziału*.
We thought that it's an appropriate name for a software which classifies 
sequences to different taxonomic units. In English the word *tiara* usually 
refers to a papal tiara. A papal tiara has three crowns, and life has three domains,
so maybe that's another explanation for the name of our program.

