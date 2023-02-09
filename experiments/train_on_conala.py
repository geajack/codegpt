from util import *
from data import CodeGPTDataset, preprocess_code_train, preprocess_train
from model import get_gpt2_tokenizer
from data.formats import conala
from train import train



if __name__ == "__main__":
    root = output_directory("train_on_conala_see_dataset")

    tokenizer = get_gpt2_tokenizer("microsoft/CodeGPT-small-py-adaptedGPT2")

    preprocessed = (
        preprocess_train(nl, code, tokenizer=tokenizer)
        for nl, code in conala("datasets/conala/train.json")
    )

    dataset = CodeGPTDataset.from_preprocessed(
        preprocessed_data=preprocessed,
        tokenizer=tokenizer
    )

    dataset.save_debug(root / "dataset.md")

    # train(
    #     model="microsoft/CodeGPT-small-py-adaptedGPT2",
    #     dataset=dataset,
    #     output_directory=root / "model"
    # )