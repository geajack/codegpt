from util import *
from data import *
from model import get_gpt2_tokenizer
from data.formats import conala
from predict import predict



if __name__ == "__main__":
    root = output_directory("evaluate_on_conala")

    tokenizer = get_gpt2_tokenizer("microsoft/CodeGPT-small-py-adaptedGPT2")

    preprocessed = (
        preprocess_test(nl, tokenizer=tokenizer)
        for nl, code in conala("datasets/conala/train.json")
    )

    dataset = CodeGPTDataset.from_preprocessed(
        preprocessed_data=preprocessed,
        tokenizer=tokenizer
    )

    outputs = predict(
        model="/home/ICTDOMAIN/d20126116/Code/CodeGPT/results/train_on_conala-08-02-23@22:19:14/model/checkpoint-last",
        dataset=dataset
    )

    save_output(root / "output", outputs)