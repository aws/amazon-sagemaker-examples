import pandas as pd

def get_dataset(filename):
    labels = []
    sentences = []
    max_length = -1
    n_rows = 0
    n_rows_limit = 5000
    with open(filename) as f:
        for line in f:
            if n_rows > n_rows_limit:
                break
            tokens = line.split()
            label = int(tokens[0])
            words = tokens[1:]
            max_length = max(max_length, len(words))
            labels.append(label)
            sentence = " ".join(words)
            sentences.append(sentence)
            n_rows += 1
        df = pd.DataFrame(list(zip(labels, sentences)),
               columns =["label","sentence"])
        
    return df

train_df = get_dataset("./data/sst2.train")
test_df = get_dataset("./data/sst2.test")
val_df = get_dataset("./data/sst2.val")


# save datasets as json for uploading to s3
train_df.to_json(f"./data/train.json", orient="records", lines=True)
test_df.to_json(f"./data/test.json", orient="records", lines=True)
val_df.to_json(f"./data/val.json", orient="records", lines=True)