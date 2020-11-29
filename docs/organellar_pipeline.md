# Organellar genomes extraction

The goal of this pipeline is to extract the organellar fraction (mitochondria and/or plastids) 
from (meta)genomic datasets and then to assemble complete organellar genomes,
which would be used for phylogenomic analyses. 

Unfortunately, organellar fraction is minority in metagenomic data, which results in low coverage and short contigs.
To overcome that we propose to separate organellar fraction and process them separately using following pipeline:

### The pipeline

1. Classify assembled contigs with **tiara**. 
2. Take the organellar fraction.
3. Extract reads that map to organellar fraction.
4. Assemble the reads again using Spades.
5. Bin them using suitable software (metabat2/concoot). # we need to check if it works
6. Use **tiara** again for separate mitochondrions from plastids.
7. Further annotate organellar MAGs. 



[Back to README](README.md)