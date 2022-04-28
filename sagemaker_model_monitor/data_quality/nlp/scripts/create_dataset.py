import wget
import os
from zipfile import ZipFile
import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import exists

file_name = 'cola_public_1.1.zip'
test_split_size = 0.15
val_split_size = 0.10
cola_dataset_url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'
if (exists(path_to_file)):
    print('CoLA dataset zip file already downloaded')
else: 
    print('Downloading CoLA dataset zip file')
    wget.download(cola_dataset_url)
    
dataset_split = 'cola'
dataset_name = 'bert'

    
with ZipFile(file_name, 'r') as zip:
    print('Extracting CoLA dataset zip file')
    zip.extractall()
    print('Done!')

df = pd.read_csv(
    "./cola_public/raw/in_domain_train.tsv",
    sep="\t",
    header=None,
    usecols=[3, 1],
    names=["label", "sentence"],
)


print('Spliting dataset into train, test and validation files')

train,test = train_test_split(df, test_size = test_split_size)
train,val = train_test_split(train, test_size = val_split_size)

print('Saving files in JSON lines record format')

train.to_json(f"./data/train.json", orient="records", lines=True)
test.to_json(f"./data/test.json", orient="records", lines=True)
val.to_json(f"./data/validation.json", orient="records", lines=True)

print('Dataset created successfully')

if os.path.exists(file_name):
    os.remove(file_name)
    print('CoLA dataset zip file deleted')
else:
    print("The file does not exist")