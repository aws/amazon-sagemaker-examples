# create data dir for wikipedia 
DATA_DIR=~/data/bert

mkdir -p DATA_DIR

wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-abstract.xml.gz -P $DATA_DIR

gunzip -c $DATA_DIR/enwiki-latest-abstract.xml.gz > $DATA_DIR/enwiki-latest-abstract.xml


