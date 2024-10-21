# LSAT

This repository contains the implementation for our CIKM 2024 paper "Preliminary Study on Incremental Learning for Large Language Model-based Recommender Systems" based on PyTorch. The repository provides code to run experiments related to incremental learning on the Amazon Book and ML-1M datasets.

## Requirement
To use this code, ensure you have the following prerequisites:
- Python 3.10
- PyTorch 1.13.1
- numpy
- pandas
- Hugging Face Transformers 4.31.0
- PEFT 0.4.0
- bitsandbytes 0.39.0
- etc..


## Project description
This repository contains the following files:
- config.py: Defines paths for LLM, and datasets. Update this file to set paths for your environment.
- train.py: Main script for model training, supporting both full training and incremental fine-tuning.
- evaluate.py: Script for model evaluation.
- fine_tune.sh: Shell script for fine-tuning models incrementally.
- full_retrain.sh: Shell script for full retraining of models from scratch.
- evaluate_retrain.sh: Shell script for evaluating models after full retraining. It runs the evaluation process on the test set to measure the performance of a fully retrained model.
- evaluate_tune.sh: Shell script for evaluating models after fine-tuning. It tests the performance of models that have undergone incremental fine-tuning using the specified evaluation metrics.
- arithmetic.py: Implements LoRA Fusion, enabling parameter-efficient fine-tuning by merging multiple LoRA layers during incremental learning.
- utils.py: Provides utility functions and support code used across the project.
- process_amazon_book_dataset.ipynb: Jupyter notebook for processing the Amazon Book dataset.
- process_ML_dataset.ipynb: Jupyter notebook for preprocessing other ML datasets.

## Datasets
The following datasets are used in this project. Download the datasets from the given links and preprocess them using the provided notebooks:

- Amazon Book Dataset
Download the dataset from the Amazon Review Data https://nijianmo.github.io/amazon/index.html. This dataset includes a variety of product reviews, including those for books, and is useful for training large-scale recommender systems.

After downloading, use the process_amazon_book_dataset.ipynb notebook to preprocess the dataset.

- MovieLens Dataset (1M)
Download the MovieLens 1M dataset from MovieLens https://grouplens.org/datasets/movielens/1m/. This dataset includes 1 million ratings of movies from various users and is a popular benchmark for recommendation system tasks.

After downloading, use the process_ML_dataset.ipynb notebook to preprocess the dataset.


## Citation

ACM ref:

>Tianhao Shi, Yang Zhang, Zhijian Xu, Chong Chen, Fuli Feng, Xiangnan He, and Qi Tian. 2024. Preliminary Study on Incremental Learning for Large Language Model-based Recommender Systems. ACM CIKM '24. https://doi.org/10.1145/3627673.3679922
