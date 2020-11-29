# Eukaryotic nuclear genomes extraction


If you want to extract eukaryotic MAGs (Metagenome assembled genomes) from metagenomic data, 
we have a pipeline presented below. The best way to reduce the complexity of metagenomic data is 
to isolate the eukaryotic fraction.

### The pipeline

1. Classify assembled contigs with **tiara**.
2. Take eukaryotic fraction *and* the [unknowns](#unknowns) (and possibly [organelles](#organelles)).
3. Extract reads that map to organellar fraction (you can skip this step).
4. Assemble again using Spades (you can skip this step).
5. Map reads to get the information about the of coverage of your fragments.
6. Bin the reads using suitable software (metabat2/concoot).
7. Estimate the genome completness using Busco.
8. Further process eukaryotic MAGs. You can use MetaEuk for instance.


#### Unknowns
To maximize completeness of your MAGs we highly recommend to add contigs of unknown origin to the process. 

We are aware of (low complex, high diverse regions, small reference from microbial eukaryotes). # redakcja

Even though possible prokaryotic contaminants should form separate bins. 

#### Organelles 
The organellar fractions could improve the quality of assembly.
The reason is that there exist multiple transfers of genes from plastids to nucleus.


[Back to README](README.md)