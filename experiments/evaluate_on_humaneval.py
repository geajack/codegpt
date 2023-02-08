from data import CodeGPTDataset, preprocess_code_only
from model import get_gpt2_tokenizer
from util import *
from data.humaneval import human_eval_contexts
from predict import predict


if __name__ == "__main__":
    model = "/home/ICTDOMAIN/d20126116/Code/CodeGPT/results/contextual_conala-07-02-23@17:51:14/model/checkpoint-last"

    root = output_directory("evaluate_on_humaneval_contextual_conala")

    tokenizer = get_gpt2_tokenizer("microsoft/CodeGPT-small-py-adaptedGPT2")

    preprocessed = (
        preprocess_code_only(code, tokenizer=tokenizer)
        for code in human_eval_contexts()
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