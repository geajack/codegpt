from util import *
from data import CodeGPTDataset, preprocess_code_train
from model import get_gpt2_tokenizer
from data.contextual_conala import contextual_conala_contexts
from train import train



if __name__ == "__main__":
    root = output_directory("contextual_conala")

    tokenizer = get_gpt2_tokenizer("microsoft/CodeGPT-small-py-adaptedGPT2")

    preprocessed = (
        preprocess_code_train(code, tokenizer=tokenizer)
        for code in contextual_conala_contexts()
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