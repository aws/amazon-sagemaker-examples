# extract 10000 lines of the wikipedia abstract
DATA_DIR=~/data/bert

cat $DATA_DIR/enwiki-lastest-abstract.xml | head -10000 > $DATA_DIR/enwiki-latest-abstract-10000.xml