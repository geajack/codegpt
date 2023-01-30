import json
from data.formats import conala

def conala_noisy(mined_filepath, train_filepath, threshold):
    yield from (1 for _ in conala(train_filepath))

    with open(mined_filepath) as file:
        lines = file.readlines()

    for line in lines:
        entry = json.loads(line)
        probability = entry["prob"]
        if probability > threshold:
            yield probability
        else:
            break

if __name__ == "__main__":
    print("threshold,n_data,percent_clean")
    for threshold in [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]:
        n = 0
        average_correct = 0
        for p in conala_noisy("datasets/conala/mined.jsonl", "datasets/conala/train.json", threshold / 100):
            n += 1
            average_correct += p

        print(threshold, n, f"{(average_correct / n) * 100:.3f}", sep=",")