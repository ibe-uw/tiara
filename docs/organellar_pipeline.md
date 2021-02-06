# Organellar fraction recovery using Tiara

Here we present pipeline for analyzing organellar fraction from metagenomes. 
In the best case using this pipeline it is possible
to recover partially complete plastid and mitochondrial genomes. 

### The pipeline

1. Classify assembled contigs with **tiara**. 
2. Take the organellar fraction with or without unknowns.
3. Extract reads that map to organellar fraction.
4. Assemble the reads again using metaSpades.
5. Bin them using suitable software (like metabat2 or concoot). 
6. Use **tiara** again for separate mitochondrions from plastids.

After that you can manually annotate formed MAGs. 

#### Minimum sequence length

If you are trying identify parts of organellar genomes of rare micro 
eukaryotes you can decrease cut-off to 1kb in your run (`--min_len 1000`),
at the expense of the increase of false positives. 

#### Unknowns

To maximize sensitivity of identification organelles, you may 
consider to add this fraction to your analysis. 
Other sequences which might end up in the class 'unknowns' 
can be easily removed during preprocessing step like binning and bin refinement.


[Back to README](README.md)
