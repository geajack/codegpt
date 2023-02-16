from data.formats import conala
from data import *
from model import get_gpt2_tokenizer

if __name__ == "__main__":
    tokenizer = get_gpt2_tokenizer("microsoft/CodeGPT-small-py-adaptedGPT2")

    preprocessed = (
        preprocess(nl, code, padding=False, tokenizer=tokenizer)
        for nl, code in conala("datasets/conala/train.json")
    )

    lengths = []
    for tokens, labels in preprocessed:
        lengths.append(len(tokens))

    mean_length = sum(lengths) / len(lengths)
    
    print(mean_length)