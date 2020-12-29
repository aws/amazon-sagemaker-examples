import os

DATA_DIR='/home/ubuntu/data/bert'
FILE = 'enwiki-latest-abstract.xml'

line = 1
with open(os.path.join(DATA_DIR, FILE), 'r') as f:
    while line < 100:
        l = f.readline()
        line +=1
        if l == '<doc>\n':
            print('doc')
        if l == '</doc>\n':
            print('endof doc')
            
        if l == '<feed>\n':
            print('feed')