# Prokaryotic fraction recovery using Tiara

If you are more interested in prokaryotic fraction 
(for example during investigating bacterial endosymbionts of eukaryotes)
we provide pipeline presented below:

### The pipeline

1. Classify assembled contigs with **tiara**.
2. Take only sequences classified as prokaryota (bacteria, archea and prokaryota classes).
3. Extract reads that map to prokaryotic fraction (you can skip this step). 
4. Assemble again using Spades (you can skip this step). 
5. Bin them using suitable software (metabat2/concoot).
6. Use **tiara** again for preliminary assessment of achieved bins (optional).
7. You can manually refine bins using Anvi'O. Further check quality of  bins can be done using checkM. We strongly reccomend to assign lineage to the bins using gtdb-tk. 

[Back to README](README.md)