# A joint embedding of protein sequence and structure enables robust variant effect predictions

## Introduction
This repository contains scripts and data to repeat the analyses in Blaabjerg et al.:
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
Data related to the paper can be download here: [https://zenodo.org/records/10362251](https://zenodo.org/records/10362251).<br>
The directory contains the folding subdirectories:<br>
* `train`
    * `model_weights`: Final weights for the SSEmb-MSATransformer and SSEmb-GVPGNN modules.
    * `optimizer_weights`: Parameters for the optimizer at time of early-stopping.
    * `msa`: MSAs for the proteins in the training set.
* `mave_val`:
    * `msa`: MSAs for the proteins in the MAVE validation set.
* `rocklin`:
    * `msa`: MSAs for the proteins in the mega-scale stability change test set.
* `proteingym`:
    * `structure`: AlphaFold-2 generated structures used for the ProteinGym test.
    * `msa`: MSAs for the proteins in the ProteinGym test set.
* `scannet`:
    * `model_weights`: Final weights for the SSEmb downstream model trained on the ScanNet data set.
    * `optimizer_weights`: Parameters for the optimizer at time of early-stopping.
    * `msa`: MSAs for the proteins in the ScanNet data set.

## License
Source code and model weights are licensed under the MIT License.

## Acknowledgements
Code for the MSA Transformer was developed by the ESM team at Meta Research:<br>
[https://github.com/facebookresearch/esm](https://github.com/facebookresearch/esm).
<br/><br/>
Code for the GVP-GNN was developed by Jing et al:<br>
[https://github.com/drorlab/gvp-pytorch](https://github.com/drorlab/gvp-pytorch).

## Citation
Please cite:
XXX

