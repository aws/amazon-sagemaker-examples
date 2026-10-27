import dgl


class HeteroGraphNeighborSampler:
    """Neighbor sampler on heterogeneous graphs
    Parameters
    ----------
    g : DGLHeteroGraph
        Full graph
    category : str
        Category name of the seed nodes.
    nhops : int
            Number of hops to sample/number of layers in the node flow.
    fanout : int
        Fanout of each hop starting from the seed nodes. If a fanout is None,
        sample full neighbors.
    """
    def __init__(self, g, category, nhops, fanout=None):
        self.g = g
        self.category = category
        self.fanouts = [fanout] * nhops

    def sample_block(self, seeds):
        blocks = []
        seeds = {self.category: seeds}
        cur = seeds
        for fanout in self.fanouts:
            if fanout is None:
                frontier = dgl.in_subgraph(self.g, cur)
            else:
                frontier = dgl.sampling.sample_neighbors(self.g, cur, fanout)
            block = dgl.to_block(frontier, cur)
            cur = {}
            for ntype in block.srctypes:
                cur[ntype] = block.srcnodes[ntype].data[dgl.NID]
            blocks.insert(0, block)
        return blocks, cur[self.category]


class NeighborSampler:
    """Neighbor sampler on homogenous graphs
    Parameters
    ----------
    g : DGLGraph
        Full graph
    nhops : int
        Number of hops to sample/number of layers in the node flow.
    fanout : int
        Fanout of each hop starting from the seed nodes. If a fanout is None,
        sample full neighbors.
        """
    def __init__(self, g, nhops, fanout=None):
        self.g = g
        self.fanouts = [fanout] * nhops
        self.nhops = nhops

    def sample_block(self, seeds):
        blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            if fanout is None:
                frontier = dgl.in_subgraph(self.g, seeds)
            else:
                frontier = dgl.sampling.sample_neighbors(self.g, seeds, fanout, replace=False)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]

            blocks.insert(0, block)
        return blocks, blocks[0].srcdata[dgl.NID]


class FullGraphSampler:
    """Does nothing and just returns the full graph
    Parameters
    ----------
    g : DGLGraph
        Full graph
    nhops : int
        Number of hops to sample/number of layers in the node flow.
    """

    def __init__(self, g, nhops):
        self.g = g
        self.nhops = nhops

    def sample_block(self, seeds):
        return [self.g] * self.nhops, seeds
