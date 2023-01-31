from data import CodeGPTDataset
from model import get_gpt2
from util import *
from data.humaneval import human_eval_prompts
from predict import predict


if __name__ == "__main__":
    model = "/home/ICTDOMAIN/d20126116/Code/CodeGPT/results/seeded_noisy_conala-22-01-23@22:59:46/models/threshold_1.0/seed_1/checkpoint-last"

    root = output_directory("evaluate_on_humaneval")

    prompts = list(human_eval_prompts())
    print(prompts[129])

    model, tokenizer = get_gpt2(model)
    dataset = CodeGPTDataset.from_test_data(
        datasource=human_eval_prompts(),
        tokenizer=tokenizer
    )

    decoded = tokenizer.decode(dataset[129][1])
    print(decoded)
    print(len(dataset[129][1]))
    print(len(decoded))