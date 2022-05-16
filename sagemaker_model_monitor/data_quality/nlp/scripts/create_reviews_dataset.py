import datasets
from datasets import load_dataset

# configuration
dataset_name = "amazon_us_reviews"
dataset_split = "Apparel_v1_00"
train_dataset_length = 5_000
test_split_size = 0.15


# load dataset using datasets library, using the Apperal split.
dataset = load_dataset("amazon_us_reviews", "Apparel_v1_00")

# since there is only a "tran" split assign it as dataset
dataset = dataset["train"]
print(f"total dataset contains: {len(dataset)} rows")


# remove unnecessary columns from dataset
remove_columns = [
    "marketplace",
    "customer_id",
    "review_id",
    "product_id",
    "product_title",
    "product_category",
    "helpful_votes",
    "total_votes",
    "product_parent",
    "vine",
    "verified_purchase",
    "review_headline",
    "review_date",
]
dataset = dataset.remove_columns(remove_columns)

# rename columns to match schema
dataset = dataset.rename_column("review_body", "review")
dataset = dataset.rename_column("star_rating", "label")
print(f"The dataset features are now {list(dataset.features.keys())}")


# shuffle dataset and select x samples
sampled_dataset = dataset.shuffle().select(range(train_dataset_length))
print(f"sampled dataset contains: {len(sampled_dataset)} rows")

# Label needs to start with 0
def index_label(example):
    if example["label"] > 3:
        example["label"] = 1
    else:
        example["label"] = 0
    return example


sampled_dataset = sampled_dataset.map(index_label)

# split sampled dataset into test and train split
processed_dataset_dict = sampled_dataset.train_test_split(test_size=test_split_size)
print(f"train dataset contains: {len(processed_dataset_dict['train'])} rows")
print(f"test dataset contains: {len(processed_dataset_dict['test'])} rows")


# save datasets as json for uploading to s3
processed_dataset_dict["train"].to_json(f"./data/train.json")
processed_dataset_dict["test"].to_json(f"./data/test.json")