if __name__ == "__main__":
    from sys import stdin, argv
    import json
    from run import MODEL_CLASSES

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

    input = stdin.readline().strip()
    if argv[1] == "--encode":
        output = tokenizer.encode(input)
        print(output)
    elif argv[1] == "--decode":
        tokens = json.loads(input)
        vocab = tokenizer.get_vocab()
        output = [tokenizer.decode(token).strip() for token in tokens]
        print(*output, sep="|")