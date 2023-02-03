# CodeGPT

This is a modified version of the CodeGPT code from [CodeXGLUE](https://github.com/microsoft/CodeXGLUE). The code is substantially copied from that project, but cleaned up to be more convenient to train/test on different datasets. This repo also has multiple standard datasets already in it.

## Set-up

### Requirements
```
pip install torch
pip install transformers
```

Note that `torch` must be installed before `transformers` so that the `transformers` library is automatically configured to use it.

### Datasets
To download the datasets you will need to have `git lfs` installed (e.g. `sudo apt install git-lfs`) and set up (run `git lfs install`). If it works, you should end up with a folder called `datasets` in the root of this repository containing several large JSON files.

## Usage

### Prediction
Run `python predict.py` (if you've downloaded the datasets) for a simple example of running prediction. Note that the output produced will be basically garbage since the script uses a Microsoft pre-trained model which has seen no fine-tuning. This script show-cases the `predict()` function in `predict.py` which has the following signature:

 > `predict(model, datasource)`
 >
 > Arguments:
 > - `model`: a string, either the URI for a HuggingFace pre-trained model such as `microsoft/CodeGPT-small-py-adaptedGPT2`, or the local path to a pre-trained model directory such as is produced by the `train.train()` function in this codebase.
 > - `datasource`: a list, generator or other iterable of strings representing natural language prompts for a coding task.
 >
 > Returns:
 >
 > This function is a generator yielding strings, the model's code output for each prompt in `datasource`, in the same order.

 ### Training

 Run `python train.py` for an example of running training. This script show-cases the `train()` function in `train.py` which has the following signature:

 > `train(model, datasource, output_directory)`
 >
 > Arguments:
 > - `model`: a string, either the URI for a HuggingFace pre-trained model such as `microsoft/CodeGPT-small-py-adaptedGPT2`, or the local path to a pre-trained model directory such as is returned by this function. This model will be used as the base for training (if the model is saved locally, it will not be modified on disk).
 > - `datasource`: a list, generator or other iterable of tuples of strings in the form `(nl, code)` where `nl` is a natural language prompt and `code` a corresponding piece of code.
 > - `output_directory`: The directory where the trained weights should be saved. This function will save regular checkpoints as well as the final trained weights, so this directory itself cannot be directly passed to `predict()` as a model, rather you should pass something like `<output_directory>/checkpoint-last`.
 >
 > Returns:
 >
 > The path to the sub-directory of `output_directory` holding the final trained weights, suitable for passing to `predict()`, or to `train()` in order to pick up training from a checkpoint.