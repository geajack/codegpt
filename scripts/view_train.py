from data import CodeGPTDataset
from data.contextual_conala import contextual_conala_contexts
from model import get_gpt2


model, tokenizer = get_gpt2("microsoft/CodeGPT-small-py-adaptedGPT2")
datasource = (("", code) for code in contextual_conala_contexts())
dataset = CodeGPTDataset.from_training_data(
    datasource=datasource,
    tokenizer=tokenizer
)

index, inputs, labels = dataset[0]

text = tokenizer.decode(inputs)

print(text)