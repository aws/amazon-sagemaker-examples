import argparse
import csv
import random

parser = argparse.ArgumentParser(description="Generate sample data")
parser.add_argument("--samples", type=int, default=10000, help="Number of samples to generate")
parser.add_argument("--filename", type=str, default="samples.csv", help="Filename to use")
parser.add_argument("--debug", type=bool, default=False, help="Whether to print debug messages")
args = parser.parse_args()

question_words = {
    "itemization": [
        "itemization",
        "deduction",
        "mortgage",
        "charitable",
        "donation",
        "expense",
        "local",
        "state",
        "tax",
    ],
    "estate taxes": ["estate", "inheritance"],
    "medical": ["medical", "expense", "covid"],
    "deferments": ["deferment", "delay", "late", "payment"],
    "investments": ["investment", "401k", "403b", "ira", "capital", "gains", "losses"],
    "properties": ["properties", "rental", "investment"],
}

question_categories = list(question_words.keys())
all_words = [word for category in question_categories for word in question_words[category]]

csvfile = open(args.filename, "w")
sample_writer = csv.writer(csvfile)
sample_writer.writerow(["label", "words"])


for i in range(args.samples):
    words = set()
    specialty = random.choice(question_categories)
    r = 1
    while r > 0.2:
        if r > 0.3:
            words.add(random.choice(question_words[specialty]))
        else:
            words.add(random.choice(all_words))
        r = random.random()

    sample = []

    sample.append("category_" + specialty)

    sample.extend(words)

    if args.debug:
        print("question specialty: ", specialty)
        print(words)
        print(sample)
        print()

    sample_writer.writerow(sample)
