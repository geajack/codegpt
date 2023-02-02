import datetime
from pathlib import Path

from util import *
from data.formats import conala
from data.contextual_conala import contextual_conala_contexts
from train import train

if __name__ == "__main__":
    root = output_directory("contextual_conala")
    train(
        model="microsoft/CodeGPT-small-py-adaptedGPT2",
        datasource=(("", code) for code in contextual_conala_contexts()),
        output_directory=root / "model"
    )