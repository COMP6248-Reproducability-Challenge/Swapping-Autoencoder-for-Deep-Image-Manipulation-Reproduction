# Reimplementation of Swapping Autoencoder for Deep Image Manipulation 2020
Reproduction of the Park et al. [Swapping Autoencoder for Deep Image Manipulation](https://arxiv.org/pdf/2007.00653v1.pdf) 2020 Paper

Re-implementers: [Callum Newlands](mailto:cn2g18@soton.ac.uk), [Jacob Scott](mailto:js11g18@soton.ac.uk), [Muhammed Qaid](mailto:mq1g18@soton.ac.uk) | University of Southampton [[ECS](https://www.ecs.soton.ac.uk/)]

## Scope of Reimplementation

This project aimed to re-implement the Swapping Autoencoder model architecture, as well as training the model on 
(a downscaled version of) the LSUN Church dataset.

## Results and Report
The full re-implementation report is available [here](https://github.com/COMP6248-Reproducability-Challenge/Swapping-Autoencoder-for-Deep-Image-Manipulation-Reproduction/blob/main/report.pdf)

### Reconstructions
<img alt="Examples of reconstructions" src="https://github.com/COMP6248-Reproducability-Challenge/Swapping-Autoencoder-for-Deep-Image-Manipulation-Reproduction/blob/main/results/recon54.jpg?raw=true" width="256">

### Swapping Modifications
<img alt="Examples of swapping manipulations" src="https://github.com/COMP6248-Reproducability-Challenge/Swapping-Autoencoder-for-Deep-Image-Manipulation-Reproduction/blob/main/results/swap33-labelled.jpg?raw=true" width="400">

### Manipulation Vector Modifications
<img alt="Examples of manipulation vector interpolation manipulations" src="https://github.com/COMP6248-Reproducability-Challenge/Swapping-Autoencoder-for-Deep-Image-Manipulation-Reproduction/blob/main/results/day-night.jpg?raw=true" width="500">

## Prerequisites

* [Anaconda](https://www.anaconda.com/) 
  * The required packages for the conda environment are specified in the ```spec_file.txt``` and ```environment.yml``` files.
* CUDA-enabled GPU (can be ran on CPU will just be much slower)
* LSUN Church dataset (Avaiable from [https://github.com/fyu/lsun](https://github.com/fyu/lsun))
  * Needs to be downloaded and unzipped and the ```dir_path``` parameter in ```data_loading.py``` needs updated.
  E.g.: ```dir_path="../data/lsun/church_outdoor_train_lmdb"```

## Installation and Usage

The project is designed to run inside a conda environment (see Prerequisites). Once you have created the environment run ```python training.py <start_iteration>``` to train the model.
(For reference, full training took 5 days on 4 Nvidia RTX 8000 GPU cards with 48GB RAM).

Alternatively a model ```run.slurm``` file has been provided for running on the ecsall partition of the IRIDIS 5 Compute Cluster. 
(This file will require significant modifications to run elsewhere).

### Pretrained Model
The pretrained model (```optimiser.pt```) is too large for GitHub (370MB), for a copy of this please contact the re-implementing authors.

## Code Structure
* ```/stylegan2_pytorch``` Local copy of [https://github.com/rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch)
* ```/taesung_data_loading``` Local copy of the data loading code from [https://github.com/taesungp/swapping-autoencoder-pytorch](https://github.com/taesungp/swapping-autoencoder-pytorch)
* ```data_loading.py``` The code to load the downloaded LSUN Church dataset in the required format
* ```decoder.py``` The decoder module
* ```encoder.py``` The encoder (generator) module
* ```experiments.py``` The code to generate the report results and figures
* ```patch_discriminator.py``` The patch discriminator module
* ```swapping_autoencoder.py``` The combined swapping autoencoder model and loss functions
* ```training.py``` The main application entry point: initialises, loads, trains, and saves the model 
