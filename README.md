# A joint embedding of protein sequence and structure enables robust variant effect predictions

## Introduction
This repository contains scripts and data to repeat the analyses in Blaabjerg et al.:<br>
[*"A joint embedding of protein sequence and structure enables robust variant effect predictions"*](https://www.biorxiv.org/content/10.1101/2023.12.14.571755v1).

## Execution
Execute the pipeline using `src/run_pipeline.py`.<br>
This main script will call other scripts in the `src` directory to train, validate and test the SSEmb model as described in the paper.

## Requirements
The code has been developed and tested in a Unix environment using the following packages:<br>
* `python==3.7.16`
* `pytorch==1.13.1`
* `pyg==2.2.0`
* `pytorch-scatter==2.1.0`
* `pytorch-cluster==1.6.0`
* `fair-esm==2.0.0`
* `numpy==1.21.6`
* `pandas==1.3.5` 
* `biopython==1.79`
* `openmm==7.6.0`
* `pdbfixer==1.8.1`
* `scipy==1.7.3`
* `scikit-learn==1.0.2`
* `tqdm==4.64.1`
* `pytz==2022.7`
* `matplotlib==3.2.2` 
* `mpl-scatter-density==0.7` 

## Downloads 
Data related to the paper can be download here: [https://zenodo.org/records/12798019](https://zenodo.org/records/12798019).<br>
The `data` directory contains the folding subdirectories:<br>
* `train`
    * `model_weights`: Final weights for the SSEmb-MSATransformer and SSEmb-GVPGNN modules.
    * `optimizer_weights`: Parameters for the optimizer at time of early-stopping.
    * `msa`: MSAs for the proteins in the training set.
* `mave_val`:
    * `msa`: MSAs for the proteins in the MAVE validation set.
* `rocklin`:
    * `msa`: MSAs for the proteins in the mega-scale stability change test set.
* `proteingym`:
    * `structure`: AlphaFold-2 generated structures used for the ProteinGym test set.
    * `msa`: MSAs for the proteins in the ProteinGym test set.
* `scannet`:
    * `model_weights`: Final weights for the SSEmb downstream model trained on the ScanNet data set.
    * `optimizer_weights`: Parameters for the optimizer at time of early-stopping.
    * `msa`: MSAs for the proteins in the ScanNet data set.
* `clinvar`:
    * `structure`: AlphaFold-2 generated structures used for the ClinVar test set.
    * `msa`: MSAs for the proteins in the ClinVar test set.

## SSEmbLab webserver
We have created an online Colab-based webserver for making SSEmb predictions called SSEmbLab. The webserver can be accessed [here](https://colab.research.google.com/github/KULL-Centre/_2023_Blaabjerg_SSEmb/blob/main/SSEmbLab.ipynb).

## License
Source code and model weights are licensed under the MIT License.

## Acknowledgements
We thank Milot Mirdita and the rest of the ColabFold Search team for help in setting up the Colab SSEmb webserver.<br>
<br>
Code for the original MSA Transformer was developed by the ESM team at Meta Research:<br>
[https://github.com/facebookresearch/esm](https://github.com/facebookresearch/esm).
<br/><br/>
Code for the original GVP-GNN was developed by Jing et al:<br>
[https://github.com/drorlab/gvp-pytorch](https://github.com/drorlab/gvp-pytorch).

## Citation
Please cite:

*Lasse M. Blaabjerg, Nicolas Jonsson, Wouter Boomsma, Amelie Stein, Kresten Lindorff-Larsen (2023). A joint embedding of protein sequence and structure enables robust variant effect predictions. bioRxiv, 2023.12.*

```
@article {Blaabjerg2023.12.14.571755,
	author = {Lasse M. Blaabjerg and Nicolas Jonsson and Wouter Boomsma and Amelie Stein and Kresten Lindorff-Larsen},
	title = {A joint embedding of protein sequence and structure enables robust variant effect predictions},
	elocation-id = {2023.12.14.571755},
	year = {2023},
	doi = {10.1101/2023.12.14.571755},
	URL = {https://www.biorxiv.org/content/early/2023/12/16/2023.12.14.571755},
	eprint = {https://www.biorxiv.org/content/early/2023/12/16/2023.12.14.571755.full.pdf},
	journal = {bioRxiv}
}
```
