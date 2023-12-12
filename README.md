# A joint embedding of protein sequence and structure enables robust variant effect predictions

## Introduction
This repository contains scripts and data to repeat the analyses in Blaabjerg et al.:
XXX

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

## Execution
Execute the pipeline using `src/run_pipeline.py`.<br>
This main script will call other scripts in the `src` directory to train, validate and test the SSEmb model as described in the paper.

## Citation
Please cite:
XXX

## License
Source code and model weights are licensed under the MIT License.

## Acknowledgements
Code for the MSA Transformer was developed by the ESM team at Meta Research:<br>
[https://github.com/facebookresearch/esm](https://github.com/facebookresearch/esm).
<br/><br/>
Code for the GVP-GNN was developed by Jing et al:<br>
[https://github.com/drorlab/gvp-pytorch](https://github.com/drorlab/gvp-pytorch).

