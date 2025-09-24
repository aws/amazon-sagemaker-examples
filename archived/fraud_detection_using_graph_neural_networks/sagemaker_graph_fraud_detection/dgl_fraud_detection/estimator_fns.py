import os
import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--nodes', type=str, default='features.csv')
    parser.add_argument('--target-ntype', type=str, default='TransactionID')
    parser.add_argument('--edges', type=str, default='relation*')
    parser.add_argument('--heterogeneous', type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        default=True, help='use hetero graph')
    parser.add_argument('--no-features', type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        default=False, help='do not use node features')
    parser.add_argument('--mini-batch', type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        default=True, help='use mini-batch training and sample graph')
    parser.add_argument('--labels', type=str, default='tags.csv')
    parser.add_argument('--validation-data', type=str, default='validation.csv')
    parser.add_argument('--new-accounts', type=str, default='test.csv')
    parser.add_argument('--predictions', type=str, default='preds.csv', help='file to save predictions on new-accounts')
    parser.add_argument('--compute-metrics', type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        default=True, help='compute evaluation metrics after training')
    parser.add_argument('--threshold', type=float, default=0, help='threshold for making predictions, default : argmax')
    parser.add_argument('--model', type=str, default='rgcn', help='gnn to use. options: gcn, graphsage, gat, gem')
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--n-epochs', type=int, default=20)
    parser.add_argument('--n-neighbors', type=int, default=10, help='number of neighbors to sample')
    parser.add_argument('--n-hidden', type=int, default=16, help='number of hidden units')
    parser.add_argument('--n-layers', type=int, default=3, help='number of hidden layers')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight for L2 loss')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout probability, for gat only features')
    parser.add_argument('--attn-drop', type=float, default=0.6, help='attention dropout for gat/gem')
    parser.add_argument('--num-heads', type=int, default=4, help='number of hidden attention heads for gat/gem')
    parser.add_argument('--num-out-heads', type=int, default=1, help='number of output attention heads for gat/gem')
    parser.add_argument('--residual', action="store_true", default=False, help='use residual connection for gat')
    parser.add_argument('--alpha', type=float, default=0.2, help='the negative slop of leaky relu')
    parser.add_argument('--aggregator-type', type=str, default="gcn", help="graphsage aggregator: mean/gcn/pool/lstm")
    parser.add_argument('--embedding-size', type=int, default=360, help="embedding size for node embedding")

    return parser.parse_args()


def get_logger(name):
    logger = logging.getLogger(name)
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger.setLevel(logging.INFO)
    return logger