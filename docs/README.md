# tiara


**tiara** - a tool for DNA sequence classification.

Classifies sequences to seven classes: 
- archea
- bacteria
- prokarya (if it's not possible to differentiate between archea and bacteria)
- eukarya
- mitochondria
- plastids
- unknown

The sequences are classified in two stages:
- In the first stage, the sequences are classified to classes 
      archea, bacteria, prokarya, eukarya, organelle and unknown.
- In the second stage, the sequences labeled as organelle in the first stage 
      are classified to either mitochondria, plastid or unknown.

For more information on the methods used, please refer to *link do pracy*.

### Requirements

- `Python >= 3.7`
- `numpy, biopython, torch, skorch, tqdm`

### Installation

Installation instructions can be found [here](docs/detailed-installation.md).

### Usage

Sample usage can be found [here](docs/usage.md).

### Sample pipelines

Here we describe some metagenomic pipelines that utilize **tiara**. 
- [Assembling eukaryotic genomes](eukaryotic_pipeline.md)
- [Extracting organellar genomes](organellar_pipeline.md)
- [Extracting prokaryotic fraction](prokaryotic_pipeline.md)

### Name

In the Polish translation of the Harry Potter book series, the Sorting Hat
(which assigned wizards to different houses) was called *Tiara Przydziału*.
We thought that it's an appropriate name for a software which classifies 
sequences to different taxonomic units. In English the word *tiara* usually 
refers to a papal tiara. A papal tiara has three crowns, and life has three domains,
so maybe that's another explanation for the name of our program.
