# Demo Relevance Feedback

This repository provides a demo for executing various relevance feedback algorithms in the context of content-based multimedia retrieval.

## Repository Structure

- `algorithms/` - Contains Python files with implementations of the following algorithms:
  - **Rocchio**
  - **Polyadic Search**
  - **PicHunter**
  - **SVM-based Active Learning**
- `static/` - Frontend assets:
  - `demo.html` - JavaScript functions for user interaction
  - `style.css` - Webpage styling
- `utils/` - Utility functions:
  - `f_files.py` - Feature extraction and dataset handling
  - `f_process_data.py` - Preprocessing functions
  - `f_display_and_feedback.py` - Functions for user interaction
  - `functions_similarity_metrics.py` - Similarity computation
- `demo.py` - Main script for executing the algorithms and handling Flask-based communication
- `demo_functions.py` - Functions used in `demo.py`
- `preprocessing.ipynb` - Notebook with instructions for data preprocessing
- `time_exp.ipynb` - Notebook for time-based performance experiments



### 3. Download and Prepare the Dataset
1. Download the feature file **"features-clip-laion.tar.gz"** from Zenodo: [Zenodo Link](https://zenodo.org/records/8188570).  
2. Follow the detailed preprocessing steps provided in `preprocessing.ipynb`.  


## Experimental Evaluation
- The `time_exp.ipynb` notebook contains time-based performance experiments for evaluating algorithm efficiency.

