# Organellar fraction recovery using Tiara

Here we present pipeline for analyzing organellar fraction from metagenomes. In the best case using this pipeline it is possible to recover partially complete plastid and mitochondrial genomes. 

### The pipeline

1. Classify assembled contigs with **tiara**. 
2. Take the organellar fraction (with or without unkowns)
3. Extract reads that map to organellar fraction.
4. Assemble the reads again using metSpades.
5. Bin them using suitable software (like metabat2 or concoot). 
6. Use **tiara** again for separate mitochondrions from plastids.

After that you can manually annotate formed MAGs. 



[Back to README](README.md)