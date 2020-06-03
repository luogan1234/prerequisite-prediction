# prerequisite-prediction

Requirements:

- Python >= 3.5
- PyTorch >= 1.3.0
- sklearn
- pytorch_pretrained_bert

Run `bash get_data.sh` to download datasets. Dataset introduction is at `dataset/readme.md`.

Run `bash experiment_main.sh` to do main experiments.

Run `bash experiment_other.sh` to do other experiments.

Run `python stat.py` to get the experiment results.

Run `python postprocess_example.sh` to get the 10-fold ensemble predictions.