echo "check-check: setup"

apt-get update
apt-get install --allow-unauthenticated -y graphviz libgraphviz-dev pkg-config
rm -rf /var/lib/apt/lists/*

pip install -U pip
pip install -U wheel setuptools
pip install --no-cache-dir autogluon==0.1.0
pip install shap
pip install PrettyTable
pip install bokeh
pip install seaborn
pip install pygraphviz

export PATH="/opt/ml/code:${PATH}"


echo "check-check: training"
python3 train.py --feature_importance True --fit_args "{'presets': ['optimize_for_deployment']}" --init_args "{'label': 'y'}"