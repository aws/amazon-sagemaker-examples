from mxnet import gluon, nd
from dgl.nn.mxnet import GraphConv, GATConv, SAGEConv
import dgl.function as fn


class HeteroRGCNLayer(gluon.Block):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        with self.name_scope():
            self.weight = {name: gluon.nn.Dense(out_size, use_bias=False) for name in etypes}
            for child in self.weight.values():
                self.register_child(child)

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            if srctype in feat_dict:
                Wh = self.weight[etype](feat_dict[srctype])
                # Save it in graph for message passing
                G.srcnodes[srctype].data['Wh_%s' % etype] = Wh
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


class HeteroRGCN(gluon.Block):
    def __init__(self, g, in_size, hidden_size, out_size, n_layers, embedding_size, ctx):
        super(HeteroRGCN, self).__init__()
        self.g = g
        self.ctx = ctx

        # Use trainable node embeddings as featureless inputs for all non target node types.
        with self.name_scope():
            self.embed_dict = {ntype: gluon.nn.Embedding(g.number_of_nodes(ntype), embedding_size)
                               for ntype in g.ntypes if ntype != 'target'}

            for child in self.embed_dict.values():
                self.register_child(child)

            # create layers
            # input layer
            self.layers = gluon.nn.Sequential()
            self.layers.add(HeteroRGCNLayer(embedding_size, hidden_size, g.etypes))
            # hidden layers
            for i in range(n_layers - 1):
                self.layers.add(HeteroRGCNLayer(hidden_size, hidden_size, g.etypes))
            # output layer
            # self.layers.add(HeteroRGCNLayer(hidden_size, out_size, g.etypes))
            self.layers.add(gluon.nn.Dense(out_size))

    def forward(self, g, features):
        # get embeddings for all node types. for target node type, use passed in target features
        h_dict = {'target': features}
        for ntype in self.embed_dict:
            if g[0].number_of_nodes(ntype) > 0:
                h_dict[ntype] = self.embed_dict[ntype](nd.array(g[0].nodes(ntype), self.ctx))

        # pass through all layers
        for i, layer in enumerate(self.layers[:-1]):
            if i != 0:
                h_dict = {k: nd.LeakyReLU(h) for k, h in h_dict.items()}
            h_dict = layer(g[i], h_dict)

        # get target logits
        # return h_dict['target']
        return self.layers[-1](h_dict['target'])


class NodeEmbeddingGNN(gluon.Block):
    def __init__(self,
                 gnn,
                 input_size,
                 embedding_size):
        super(NodeEmbeddingGNN, self).__init__()

        with self.name_scope():
            self.embed = gluon.nn.Embedding(input_size, embedding_size)
            self.gnn = gnn

    def forward(self, g, nodes):
        features = self.embed(nodes)
        h = self.gnn(g, features)
        return h


class GCN(gluon.Block):
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
        self.layers = gluon.nn.Sequential()
        # input layer
        self.layers.add(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.add(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        # self.layers.add(GraphConv(n_hidden, n_classes))
        self.layers.add(gluon.nn.Dense(n_classes))
        self.dropout = gluon.nn.Dropout(rate=dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers[:-1]):
            if i != 0:
                h = self.dropout(h)
            h = layer(g[i], h)
        return self.layers[-1](h)


class GraphSAGE(gluon.Block):
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
            self.layers = gluon.nn.Sequential()
            # input layer
            self.layers.add(SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
            # hidden layers
            for i in range(n_layers - 1):
                self.layers.add(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
            # output layer
            self.layers.add(gluon.nn.Dense(n_classes))

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers[:-1]):
            h_dst = h[:g[i].number_of_dst_nodes()]
            h = layer(g[i], (h, h_dst))
        return self.layers[-1](h)


class GAT(gluon.Block):
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
        self.gat_layers = []
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            (in_dim, in_dim), num_hidden, heads[0],
            feat_drop, attn_drop, alpha, False))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                (num_hidden * heads[l-1], num_hidden * heads[l-1]), num_hidden, heads[l],
                feat_drop, attn_drop, alpha, residual))
        # output projection
        self.output_proj = gluon.nn.Dense(num_classes)
        for i, layer in enumerate(self.gat_layers):
            self.register_child(layer, "gat_layer_{}".format(i))
        self.register_child(self.output_proj, "dense_layer")

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h_dst = h[:g[l].number_of_dst_nodes()]
            h = self.gat_layers[l](g[l], (h, h_dst)).flatten()
            h = self.activation(h)
        # output projection
        logits = self.output_proj(h)
        return logits
