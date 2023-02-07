from data import CodeGPTDataset
from data.humaneval import human_eval_contexts
from model import get_gpt2


model, tokenizer = get_gpt2("microsoft/CodeGPT-small-py-adaptedGPT2")
datasource = human_eval_contexts()
dataset = CodeGPTDataset.from_test_data(
    datasource=datasource,
    tokenizer=tokenizer
)

index, inputs, labels = dataset[0]

text = tokenizer.decode(inputs)

print(text)