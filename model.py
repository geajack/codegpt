import torch
from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
    # OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
    # RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
    # DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer
)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    # 'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    # 'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    # 'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    # 'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}


def get_gpt2(weights):
    model_type   = "gpt2"

    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    
    tokenizer = tokenizer_class.from_pretrained(
        weights,
        do_lower_case=False,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<|UNKNOWN|>",
        sep_token="concode_elem_sep"
    )

    model = model_class.from_pretrained(weights)
    model.resize_token_embeddings(len(tokenizer))    
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer