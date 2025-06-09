# Clustered Federated Learning via Embedding Distributions

Dekai Zhang, Matt Williams, Francesca Toni

## Main Idea

EMD-CFL is a novel clustered federated learning method, which uses embedding space distribution distances to identify clusters of similar clients.

## Project Structure

The main source code is in the `src` directory. Data should be placed in a newly created `data` directory. Scripts to run each of the CFL methods can be found in the root directory, along with a `config.yaml`. Model checkpoints can be evaluated using `test.py` and the `test_config.yaml`.
