# prerequisite-prediction

Requirements:

- Python >= 3.5
- PyTorch >= 1.3.0
- sklearn
- pytorch_pretrained_bert

Process:

1. Run `bash get_data.sh` to download datasets. Dataset introduction is at `dataset/readme.md`.

2. Run `bash experiment_main.sh` to do main experiments.

3. Run `bash experiment_other.sh` to do other experiments.

4. Run `python stat.py` to get the experiment results.

5. Run `python postprocess_example.sh` to get the 10-fold ensemble predictions.