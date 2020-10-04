# prerequisite-prediction

Requirements:

- Python >= 3.5
- PyTorch >= 1.3.0
- sklearn
- transformers

Process:

1. Run `python init.py` to download datasets. Dataset introduction is at `dataset/readme.md`.

2. Run `bash experiment_main.sh` to do main experiments.

3. Run `bash experiment_other.sh` to do other experiments.

4. Run `python stat.py` to get the experiment results.

5. Run `bash postprocess.sh` to get the ensemble predictions.

Experiments run on gpu by default.