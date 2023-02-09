from pathlib import Path
import pickle
import json

import torch
from torch.utils.data import Dataset


def preprocess_code_only(code, tokenizer, block_size=512):
    code_tokens = tokenizer.encode(code)
    while len(code_tokens) > block_size:
        code_tokens = code_tokens[:-1]

    labels = [2] * len(code_tokens)
    return code_tokens, labels


def preprocess_code_train(code, tokenizer, block_size=512):
    code_tokens, labels = preprocess_code_only(code, tokenizer, block_size)
    pad_length = block_size - len(code_tokens)
    if pad_length > 0:
        code_tokens += [tokenizer.pad_token_id] * pad_length
        labels += [0] * pad_length

    return code_tokens, labels


def preprocess_train(nl, code, tokenizer, block_size=512):
    code_tokens = tokenizer.encode(code)
    nl_tokens, labels = preprocess_test(nl, tokenizer, block_size=float("inf"))

    while (len(code_tokens) + len(nl_tokens) + 2 > block_size):
        if (len(code_tokens) > len(nl_tokens)):
            code_tokens = code_tokens[:-1]
        else:
            nl_tokens = nl_tokens[:-1]

    inputs = nl_tokens + code_tokens + [tokenizer.eos_token_id]
    labels += [2] * len(code_tokens) + [0]
    assert len(inputs) <= block_size
    pad_len = block_size - len(inputs)
    inputs += [tokenizer.pad_token_id] * pad_len
    labels += [0] * pad_len
    assert len(inputs) == len(labels), (len(inputs), len(labels))

    return inputs, labels


def preprocess_test(nl, tokenizer, block_size=512):
    nl_tokens = tokenizer.encode(nl)
    while len(nl_tokens) + 2 > block_size:
        nl_tokens = nl_tokens[:-1]

    inputs = nl_tokens + [tokenizer.bos_token_id]
    labels = [1] * len(nl_tokens) + [2]

    return inputs, labels


class CodeGPTDataset(Dataset):

    @staticmethod
    def from_training_data(datasource, tokenizer, block_size=512):
        preprocessed_data = (
            preprocess_train(nl, code, tokenizer=tokenizer, block_size=block_size)
            for nl, code in datasource
        )
        return CodeGPTDataset.from_preprocessed(preprocessed_data, tokenizer)

    @staticmethod
    def from_test_data(datasource, tokenizer, block_size=512):
        preprocessed_data = (
            preprocess_test(nl, tokenizer=tokenizer, block_size=block_size)
            for nl in datasource
        )
        return CodeGPTDataset.from_preprocessed(preprocessed_data, tokenizer)

    def from_preprocessed(preprocessed_data, tokenizer):
        dataset = CodeGPTDataset(tokenizer)
        for inputs, labels in preprocessed_data:
            dataset.inputs.append(inputs)
            dataset.token_labels.append(labels)
        return dataset

    def __init__(self, tokenizer):
        self.inputs = []
        self.token_labels = []
        self.tokenizer = tokenizer
            

    def save(self, filepath):
        with open(filepath, "wb") as handle:
            pickle.dump({'inputs': self.inputs, 'token_labels': self.token_labels}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_debug(self, filepath):
        Path(filepath).parent.mkdir(exist_ok=True, parents=True)
        with open(filepath, "w") as handle:
            for tokens in self.inputs:
                decoded = self.tokenizer.decode(tokens)
                print("```", file=handle)
                print(decoded, file=handle)
                print("```", file=handle)
                print()            

    def save_tokens(self, filepath):
        with open(filepath, "w") as handle:
            for tokens in self.inputs:
                output = [tokenizer.decode(token).strip() for token in tokens]
                print(*output, sep="|", file=handle)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return item, torch.tensor(self.inputs[item]), torch.tensor(self.token_labels[item])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


if __name__ == "__main__":
    import logging
    from run import MODEL_CLASSES

    logger = logging.getLogger(__name__)

    _, _, tokenizer_class = MODEL_CLASSES["gpt2"]

    tokenizer = tokenizer_class.from_pretrained(
        "microsoft/CodeGPT-small-java-adaptedGPT2",
        do_lower_case=False,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<|UNKNOWN|>",
        sep_token="concode_elem_sep"
    )

    test_dataset = CodeGPTDataset(
        tokenizer=tokenizer,
        filepath="datasets/miniconcode/test.json",
        block_size=512,
        mode="test"
    )

    test_dataset.save("output/miniconcode_test_preprocessed.pickle")
    del test_dataset

    train_dataset = CodeGPTDataset(
        tokenizer=tokenizer,
        filepath="datasets/miniconcode/train.json",
        block_size=512,
        mode="train"
    )

    train_dataset.save("output/miniconcode_train_preprocessed.pickle")
    del train_dataset
