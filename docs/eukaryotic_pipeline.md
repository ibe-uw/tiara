# Eukaryotic MAGs recovery using Tiara.

Below we present list of instructions that lead to eukaryotic genomes from metagenomes.  

### The pipeline

1. Classify assembled contigs with **tiara**.
2. Take eukaryotic fraction *and* the [unknowns](#unknowns) (and possibly [organelles](#organelles)).
3. Extract reads that map to organellar fraction (you can skip this step).
4. Assemble again using Spades (you can skip this step).
5. Map reads to get the information about the of coverage of your fragments.
6. Bin the reads using suitable software (using for example concoot).
7. Also, you can use Anvi'O package for manual bin refinement.

sAfter that you can start analyzing your newly identified genome. The bin completness can be estimated using Busco, also we recommend to use MetaEuk for gene prediction.  


#### Unknowns
To maximize completeness of your MAGs we highly recommend to add contigs of unknown origin to the process. The prokaryotic and viral sequences which might end up in the class “unknown” can be easily removed during preprocessing step like binning and bin refinement. 

#### Organelles 
Adding organellar fraction can improve the quality of assembly. The reason is that there exist multiple transfers of genes from plastids to nucleus. 

[Back to README](README.md)