import torch

from run import set_seed, MODEL_CLASSES, update_config

def get_gpt2():
    model_type   = "gpt2"
    pretrain_dir = "microsoft/CodeGPT-small-java-adaptedGPT2"
    pretrain_dir = "../models/concode"
    local_rank = -1
    
    n_gpu = torch.cuda.device_count()

    set_seed(0)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    tokenizer = tokenizer_class.from_pretrained(
        pretrain_dir,
        do_lower_case=False,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<|UNKNOWN|>",
        sep_token="concode_elem_sep"
    )
    model = model_class.from_pretrained(pretrain_dir)
    model.resize_token_embeddings(len(tokenizer))
    update_config(model, tokenizer)

    return model, tokenizer