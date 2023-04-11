cd /opt/ml/code 
source ./venv/bin/activate

LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH python3 train_dreambooth.py "$@"