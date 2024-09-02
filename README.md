# Molecular Graph Representation Learning via Structural Similarity Information
ECML/PKDD 2024 Accpect 

## Requirements
Run ```conda env create -f environment.yml``` to create a runtime environment.


## Part 1: Molecular Structural Similarity Motif Graph Construction
Run ```python preprocess.py``` to construct an MSSM-graph for TUDataset (PROTEINS/PTC_MR/MUTAG/NCI1/Mutagenicity).

Change the parameter of drop_node() function in the ops.py to drop noises in the motif dictionary.

Run ```python preprocess_hiv.py``` and ```python preprocess_pcba.py``` to construct MSSM-graph for ogbg-molhiv and ogbg-pcba datasets.

For ogbg-pcba dataset, because 11 graphs do not have motifs, you need to subtract 11 from self.num_cliques.

## Part 2: Training and evaluation
Run ```python main.py``` for TUDataset(PROTEINS/PTC_MR/MUTAG/NCI1/Mutagenicity).

Run ```python main_ogbg_molhiv.py``` for ogbg-molhiv.

Run ```python main_molpcba.py``` for ogbg-pcba.
