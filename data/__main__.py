from config import load_dataset

if __name__ == "__main__":
    dataset = load_dataset("configs/train/conala_noisy_30.yaml")
    
    length = len(list(dataset))

    print(length)