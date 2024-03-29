from data import *
from data.contextual_conala import contextual_conala_contexts
from model import get_gpt2_tokenizer
from util import *
from predict import predict


if __name__ == "__main__":
    model = "/home/ICTDOMAIN/d20126116/Code/CodeGPT/results/train_on_contextual_conala-09-02-23@17:25:27/model/checkpoint-last"

    root = output_directory("evaluate_on_contextual_conala_from_cc")

    tokenizer = get_gpt2_tokenizer("microsoft/CodeGPT-small-py-adaptedGPT2")

    preprocessed = (
        preprocess(preamble, "", tokenizer=tokenizer, padding=False)
        for preamble, body in contextual_conala_contexts("datasets/contextual_conala/test.jsonl")
    )

    dataset = CodeGPTDataset.from_preprocessed(
        preprocessed_data=preprocessed,
        tokenizer=tokenizer
    )

    output_source = predict(
        model=model,
        dataset=dataset
    )

    save_output(
        output_file=root / f"output",
        output_source=output_source
    )