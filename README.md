# Hierarchical Diffusion Model for Climate Super-Resolution

This repository contains the Python script for training and evaluating a conditional diffusion model for statistical downscaling of climate data. The model is built upon the Elucidating the Design Space of Diffusion-Based Generative Models (EDM) framework and employs a novel hierarchical sampling strategy to generate high-resolution climate variables from coarse inputs.

Note that the hierarchical portion was added to the existing diffusion downscaling approach here: https://github.com/robbiewatt1/ClimateDiffuse (Watt & Mansfield, 2024) which was built off the original EDM model (Karras et al., 2022) 

## Key Features

- We make several additions to the existing ClimateDiffuse directory as described in the paper (Arxiv Link), which include a new approach for coarse-to-fine hierarchical modelling

## Dependencies

See the requirements.txt file for the full list of packages needed to run this file

## Usage

### 1. Data Setup

Note that the script is currently setup to read climate data from a specific directory structure. Due to the size involved with ERA5 data, we do not provide this, but note that it can be accessed through 
the API at: https://cds.climate.copernicus.eu/how-to-api. Ultimately, the model described here converts a series of .netcdf files into numpy arrays and then to torch tensors so any of the formats can be 
used as long as these are plugged into the existing architecture. 

* The script is configured to read ERA5 climate data from a specific directory structure (e.g., `/g/data/rt52/era5/single-levels/reanalysis/`).
* Static surface data (`soil_type.npy`, `topography.npy`, `land_mask.npy`) must be placed in the path specified within the `load_surface_inputs` function.
* The data loading and processing logic is handled by the custom `src.DatasetAUS` module.

### 2. Running the Script

All training, validation, and model parameters and inference are configured within the `main()` function of the script 'Train-Diffusion-HDD.py' . To run the training and evaluation process, execute the script from your terminal:

```bash
python Train-Diffusion-HDD.py
