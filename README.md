# ev-model-adapt

Work in progress. Code for a study on adapting car-following models to
ACC-controlled vehicles that were not seen during training.

## Repository layout

    config/    all settings (YAML); the only place where choices and numbers are set
    ema/       library code (package name: ema)
    scripts/   pipeline steps, run in numerical order
    tests/     tests for ema/
    results/   outputs of the pipeline (not yet included)

## Setup (Google Colab)

    pip install -r requirements.txt
    import os
    os.environ["EMA_DATA_ROOT"] = "/content/drive/MyDrive/<path to the shared data folder>"

## Data

The data are not included in this repository. The code reads the processed ACC
trajectories from the folder given by the environment variable `EMA_DATA_ROOT`
(file names, md5 checksum and expected row counts: `config/data.yaml`) and never
writes to that folder. The md5 checksum is verified before the data are used.

## Tests

    python -m pytest tests -q
