# Demo Relevance Feedback

This repository provides a demo for executing various relevance feedback algorithms in the context of content-based multimedia retrieval.

## Repository Structure

- `algorithms_functions/` - Python implementations for the following algorithms:
  - **Rocchio**
  - **Polyadic Search**
  - **PicHunter**
  - **SVM-based Active Learning**
- `static/` - Frontend assets:
  - `demo.html` - JavaScript functions for user interaction
  - `style.css` - Webpage styling
- `demo.py` - Main script invoking the algorithms and handling Flask-based communication
- `f_files.py` - Functions for extracting features and IDs from the dataset
- `preprocessing.ipynb` - Instructions for data preprocessing

## Setup and Execution

### Install Dependencies
Ensure Python is installed, then run:
```bash
pip install -r requirements.txt
```
## Download and Prepare the Dataset
Download the feature file "features-clip-laion.tar.gz" from Zenodo: https://zenodo.org/records/8188570.
Extract the features and IDs using the functions in f_files.py. Detailed preprocessing steps are provided in preprocessing.ipynb.