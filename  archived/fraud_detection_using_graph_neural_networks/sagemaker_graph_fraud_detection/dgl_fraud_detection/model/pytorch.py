import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv, GATConv, SAGEConv
import dgl.function as fn


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
                name: nn.Linear(in_size, out_size) for name in etypes
            })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            if srctype in feat_dict:
                Wh = self.weight[etype](feat_dict[srctype])
                # Save it in graph for message passing
                G.nodes[srctype].data['Wh_%s' % etype] = Wh
                # Specify per-relation message passing functions: (message_func, reduce_func).
                # Note that the results are saved to the same destination feature 'h', which
                # hints the type wise reducer for aggregation.
                funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return {ntype: G.dstnodes[ntype].data['h'] for ntype in G.ntypes if 'h' in G.dstnodes[ntype].data}


class HeteroRGCN(nn.Module):
    def __init__(self, g, in_size, hidden_size, out_size, n_layers, embedding_size):
        super(HeteroRGCN, self).__init__()
        # Use trainable node embeddings as featureless inputs.
        embed_dict = {ntype: nn.Parameter(torch.Tensor(g.number_of_nodes(ntype), in_size))
                      for ntype in g.ntypes if ntype != 'user'}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)
        # create layers
        self.layers = nn.Sequential()
        self.layers.add_module(HeteroRGCNLayer(embedding_size, hidden_size, g.etypes))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.add_module = HeteroRGCNLayer(hidden_size, hidden_size, g.etypes)

        # output layer
        self.layers.add(nn.Dense(hidden_size, out_size))

    def forward(self, g, features):
        # get embeddings for all node types. for user node type, use passed in user features
        h_dict = self.embed
        h_dict['user'] = features

        # pass through all layers
        for i, layer in enumerate(self.layers[:-1]):
            if i != 0:
                h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
            h_dict = layer(g[i], h_dict)

        # get user logits
        # return h_dict['user']
        return self.layers[-1](h_dict['user'])


class NodeEmbeddingGNN(nn.Module):
    def __init__(self,
                 gnn,
                 input_size,
                 embedding_size):
        super(NodeEmbeddingGNN, self).__init__()

        self.embed = nn.Embedding(input_size, embedding_size)
        self.gnn = gnn

    def forward(self, g, nodes):
        features = self.embed(nodes)
        h = self.gnn(g, features)
        return h


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.Sequential()
        # input layer
        self.layers.add_module(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.add(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        # self.layers.add(GraphConv(n_hidden, n_classes))
        self.layers.add(nn.Linear(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers[:-1]):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return self.layers[-1](h)


class GraphSAGE(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.g = g

        with self.name_scope():
            self.layers = nn.Sequential()
            # input layer
            self.layers.add_module(SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
            # hidden layers
            for i in range(n_layers - 1):
                self.layers.add_module(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
            # output layer
            self.layers.add_module(nn.Linear(n_hidden, n_classes))

    def forward(self, g, features):
        h = features
        for layer in self.layers[:-1]:
            h = layer(g, h)
        return self.layers[-1](h)


class GAT(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 alpha,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, alpha, False))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, alpha, residual))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, alpha, residual))

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten()
            h = self.activation(h)
        # output projection
        logits = self.gat_layers[-1](g, h).mean(1)
        return logits
