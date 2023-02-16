from util import *
from data import *
from model import get_gpt2_tokenizer
from data.contextual_conala import contextual_conala_contexts
from train import train



if __name__ == "__main__":
    root = output_directory("train_on_contextual_conala")

    tokenizer = get_gpt2_tokenizer("microsoft/CodeGPT-small-py-adaptedGPT2")

    preprocessed = (
        preprocess(preamble, body, tokenizer=tokenizer)
        for preamble, body in contextual_conala_contexts("datasets/contextual_conala/train.jsonl")
    )

    dataset = CodeGPTDataset.from_preprocessed(
        preprocessed_data=preprocessed,
        tokenizer=tokenizer
    )

    train(
        model="microsoft/CodeGPT-small-py-adaptedGPT2",
        dataset=dataset,
        output_directory=root / "model"
    )