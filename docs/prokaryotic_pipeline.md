# Prokaryotes 

If you are more interested in prokaryotic fraction 
(for example investigating bacterial endosymbionts of eukaryotic cells)
we provide pipeline presented below:

### The pipeline

1. Classify assembled contigs with **tiara**.
2. Take only sequences classified as prokaryota (bacteria, archea and prokaryota classes).
3. Extract reads that map to prokaryotic fraction (you can skip this step). 
4. Assemble again using Spades (you can skip this step). 
5. Bin them using suitable software (metabat2/concoot).
6. Use **tiara** again for preliminary assessment of achieved bins (optional).
7. Further check quality of achieved bins (for example using checkM), or assign specific taxonomy with gtdb-tk.


[Back to README](README.md)