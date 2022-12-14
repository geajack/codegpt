import pickle
import json

import torch
from torch.utils.data import Dataset


def preprocess(nl, code, tokenizer, mode, block_size=512):
    nl_tokens = tokenizer.encode(nl)
    code_tokens = tokenizer.encode(code)

    if mode == "test":
        code_tokens = []
    
    while (len(code_tokens) + len(nl_tokens) + 2 > block_size):
        if (len(code_tokens) > len(nl_tokens)):
            code_tokens = code_tokens[:-1]
        else:
            nl_tokens = nl_tokens[:-1]
    
    inputs = nl_tokens + [tokenizer.bos_token_id]
    labels = [1] * len(nl_tokens) + [2]

    if mode == "train":
        inputs += code_tokens + [tokenizer.eos_token_id]
        labels += [2] * len(code_tokens) + [0]
        assert len(inputs) <= block_size
        pad_len = block_size - len(inputs)
        inputs += [tokenizer.pad_token_id] * pad_len
        labels += [0] * pad_len
        assert len(inputs) == len(labels), (len(inputs), len(labels))

    return inputs, labels


class CodeGPTDataset(Dataset):

    def __init__(self, datasource, mode, tokenizer, block_size=512):        
        self.inputs = []
        self.token_labels = []
        self.tokenizer = tokenizer

        for nl, code in datasource:
            inputs, labels = preprocess(nl, code, tokenizer=tokenizer, mode=mode, block_size=block_size)
            self.inputs.append(inputs)
            self.token_labels.append(labels)
            

    def save(self, filepath):
        with open(filepath, "wb") as handle:
            pickle.dump({'inputs': self.inputs, 'token_labels': self.token_labels}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_debug(self, filepath):
        with open(filepath, "w") as handle:
            for tokens in self.inputs:
                output = [tokenizer.decode(token).strip() for token in tokens]
                print(*output, sep="|", file=handle)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return item, torch.tensor(self.inputs[item]), torch.tensor(self.token_labels[item])


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
