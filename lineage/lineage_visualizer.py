from __future__ import division
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from sagemaker.lineage.context import Context
from sagemaker.lineage.artifact import Artifact
from sagemaker.lineage.association import Association
from sagemaker.lineage.action import Action
from datetime import datetime


class LineageVisualizer(object):
    def __init__(self, sagemaker_client):
        self._sm_client = sagemaker_client

    def upstream(self, start_arn):
        edges = self._get_upstream_lineage(start_arn)
        self._plot_lineage(edges)

    def _get_upstream_lineage(self, start_arn):
        upstream_associations = Association.list(
            destination_arn=start_arn, sagemaker_boto_client=self._sm_client
        )
        unexplored_associations = list(upstream_associations)
        edges = []
        while unexplored_associations:
            association = unexplored_associations.pop()
            src = association.source_arn
            dest = association.destination_arn
            edges.append(association)
            upstream_associations = Association.list(
                destination_arn=src, sagemaker_boto_client=self._sm_client
            )
            unexplored_associations.extend(upstream_associations)
        return edges

    def downstream(self, start_arn):
        edges = self._get_downstream_lineage(start_arn)
        self._plot_lineage(edges)

    def _get_downstream_lineage(self, start_arn):
        downstream_associations = Association.list(
            source_arn=start_arn, sagemaker_boto_client=self._sm_client
        )
        unexplored_associations = list(downstream_associations)
        edges = []
        while unexplored_associations:
            association = unexplored_associations.pop()
            src = association.source_arn
            dest = association.destination_arn
            edges.append(association)
            downstream_associations = Association.list(
                destination_arn=src, sagemaker_boto_client=self._sm_client
            )
            unexplored_associations.extend(downstream_associations)
        return edges

    def both(self, start_arn):
        upstream = self._get_upstream_lineage(start_arn)
        downstream = self._get_downstream_lineage(start_arn)
        all = []
        if upstream:
            all.extend(upstream)
        if downstream:
            all.extend(downstream)
        self._plot_lineage(all)

    def write_yaml(self):
        file_name = f"graph_{datetime.now().timestamp()}.yaml"
        nx.write_yaml(self._g, file_name)
        return file_name

    def _plot_lineage(self, edges):
        G = nx.DiGraph()

        for edge in edges:
            source_name = edge.source_arn.split("/")[1]
            source_name = f"{edge.source_type}-({source_name})"
            G.add_node(source_name)
            dest_name = edge.destination_arn.split("/")[1]
            dest_name = f"{edge.desination_type}-({dest_name})"
            G.add_node(dest_name)
            G.add_edge(source_name, dest_name)
        self._g = G
        M = G.number_of_edges()

        pos = nx.layout.spring_layout(G)
        nodes = nx.draw_networkx_nodes(G, pos, node_size=500)
        nx.draw_networkx_labels(G, pos)
        edges = nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=30, width=1, arrows=True)

        ax = plt.gca()
        ax.patch.set_facecolor("white")
        ax.figure.set_size_inches(10, 10)
        # fig= plt.figure(figsize=(10,10))
        plt.title("foo")
        plt.show()
