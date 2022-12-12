#!/home/ICTDOMAIN/d20126116/Code/CodeGPT/pyenv/bin/python

from __future__ import absolute_import, division, print_function

import os
import pickle
import json

import torch
from torch.utils.data import Dataset


class concodeDataset(Dataset):
    def __init__(self, data_dir, tokenizer, cache_file=None, logger=None, overwrite_cache=False, local_rank=-1, file_type='train', block_size=512, mode='train'):
        if local_rank==-1:
            local_rank=0
            world_size=1
        else:
            local_rank=local_rank
            world_size=torch.distributed.get_world_size()

        self.block_size = block_size
        self.mode = mode

        try:
            with open(cache_file, 'rb') as handle:
                data = pickle.load(handle)
                self.inputs = data['inputs']
                self.token_labels = data['token_labels']
        except FileNotFoundError:
            self.inputs = []
            self.token_labels = []

            datafile = os.path.join(data_dir, f"{file_type}.json")
            if file_type == 'train':
                logger.warning("Creating features from dataset file at %s", datafile)
            datas = open(datafile).readlines()

            length = len(datas)
            logger.info("Data size: %d"%(length))
            for idx, x in enumerate(datas):
                if idx % (length//10) == 0:
                    percent = idx / (length//10) * 10
                    logger.warning("Rank %d, load %d"%(local_rank, percent))
                if idx % world_size != local_rank:
                    continue
                x = json.loads(x)
                code = tokenizer.encode(x["code"])
                nl = tokenizer.encode(x["nl"])

                input_ids, input_labels = self.pad_and_get_mask(code, nl, tokenizer)
                self.inputs.append(input_ids)
                self.token_labels.append(input_labels)

            with open(cache_file, 'wb') as handle:
                pickle.dump({'inputs': self.inputs, 'token_labels': self.token_labels}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def pad_and_get_mask(self, code, nl, tokenizer):
        if self.mode == 'test':
            code = []
        while (len(code) + len(nl) + 2 > self.block_size):
            if (len(code) > len(nl)):
                code = code[:-1]
            else:
                nl = nl[:-1]
        if self.mode == 'train':
            inputs = nl + [tokenizer.bos_token_id] + code + [tokenizer.eos_token_id]
            labels = [1] * len(nl) + [2] * (len(code)+1) + [0]
        else:
            inputs = nl + [tokenizer.bos_token_id]
            labels = [1] * len(nl) + [2]
            return inputs, labels
        assert len(inputs) <= self.block_size
        pad_len = self.block_size - len(inputs)
        inputs += [tokenizer.pad_token_id] * pad_len
        labels += [0] * pad_len
        assert len(inputs) == len(labels)
        return inputs, labels


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item]), torch.tensor(self.token_labels[item])


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

    dataset = concodeDataset(
        tokenizer=tokenizer,
        data_dir="datasets/concode",
        cache_file="output/concode_test_preprocessed.pickle",
        logger=logger,
        file_type="test",
        block_size=512,
        mode="test"
    )
