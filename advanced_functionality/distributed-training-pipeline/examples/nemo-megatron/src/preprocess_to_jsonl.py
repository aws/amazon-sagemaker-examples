from datasets import load_dataset, DatasetDict
import json

def convert_to_jsonl(dataset, path:str):
    with open(path, "w", encoding='utf-8') as f:
        for sample in dataset:
          input = f"### Instruction:\n{sample['instruction']}\n ### Input:\n{sample['input']}\n"
          output = f"### Response:\n{sample['output']}"
          json_obj = {"input": input, "output": output}
          json_string = json.dumps(json_obj, ensure_ascii=False) + "\n"
          f.write(json_string)

def load_dolphin_dataset():
    dataset = load_dataset("cognitivecomputations/dolphin", "flan1m-alpaca-uncensored", num_proc=8)
    # 90% train, 10% test + validation
    train_testvalid = dataset['train'].train_test_split(test_size=0.1)
    # Split the 10% test + valid in half test, half valid
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    # gather everyone if you want to have a single DatasetDict
    dataset = DatasetDict({
        'train': train_testvalid['train'],
        'valid': test_valid['train'],
        'test': test_valid['test']})

    print(dataset)
    return dataset

def main():
    dataset = load_dolphin_dataset()
    convert_to_jsonl(dataset['train'], "dolphin_train.jsonl")
    convert_to_jsonl(dataset['test'], "dolphin_test.jsonl")
    convert_to_jsonl(dataset['valid'], "dolphin_valid.jsonl")

if __name__ == "__main__":
    main()