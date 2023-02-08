from data import CodeGPTDataset, preprocess_code_train
from data.contextual_conala import contextual_conala_contexts
from model import get_gpt2, get_gpt2_tokenizer


tokenizer = get_gpt2_tokenizer("microsoft/CodeGPT-small-py-adaptedGPT2")

preprocessed = (
    preprocess_code_train(code, tokenizer=tokenizer)
    for code in contextual_conala_contexts()
)

dataset = CodeGPTDataset.from_preprocessed(
    preprocessed_data=preprocessed,
    tokenizer=tokenizer
)

index, inputs, labels = dataset[0]

text = tokenizer.decode(inputs)

print(text)