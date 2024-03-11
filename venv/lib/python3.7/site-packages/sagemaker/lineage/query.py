# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""This module contains code to query SageMaker lineage."""
from __future__ import absolute_import

from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union, List, Dict
from json import dumps
from re import sub, search

from sagemaker.utils import get_module
from sagemaker.lineage._utils import get_resource_name_from_arn


class LineageEntityEnum(Enum):
    """Enum of lineage entities for use in a query filter."""

    TRIAL = "Trial"
    ACTION = "Action"
    ARTIFACT = "Artifact"
    CONTEXT = "Context"
    TRIAL_COMPONENT = "TrialComponent"


class LineageSourceEnum(Enum):
    """Enum of lineage types for use in a query filter."""

    CHECKPOINT = "Checkpoint"
    DATASET = "DataSet"
    ENDPOINT = "Endpoint"
    IMAGE = "Image"
    MODEL = "Model"
    MODEL_DATA = "ModelData"
    MODEL_DEPLOYMENT = "ModelDeployment"
    MODEL_GROUP = "ModelGroup"
    MODEL_REPLACE = "ModelReplaced"
    TENSORBOARD = "TensorBoard"
    TRAINING_JOB = "TrainingJob"
    APPROVAL = "Approval"
    PROCESSING_JOB = "ProcessingJob"
    TRANSFORM_JOB = "TransformJob"


class LineageQueryDirectionEnum(Enum):
    """Enum of query filter directions."""

    BOTH = "Both"
    ASCENDANTS = "Ascendants"
    DESCENDANTS = "Descendants"


class Edge:
    """A connecting edge for a lineage graph."""

    def __init__(
        self,
        source_arn: str,
        destination_arn: str,
        association_type: str,
    ):
        """Initialize ``Edge`` instance."""
        self.source_arn = source_arn
        self.destination_arn = destination_arn
        self.association_type = association_type

    def __hash__(self):
        """Define hash function for ``Edge``."""
        return hash(
            (
                "source_arn",
                self.source_arn,
                "destination_arn",
                self.destination_arn,
                "association_type",
                self.association_type,
            )
        )

    def __eq__(self, other):
        """Define equal function for ``Edge``."""
        return (
            self.association_type == other.association_type
            and self.source_arn == other.source_arn
            and self.destination_arn == other.destination_arn
        )

    def __str__(self):
        """Define string representation of ``Edge``.

        Format:
            {
                'source_arn': 'string',
                'destination_arn': 'string',
                'association_type': 'string'
            }

        """
        return str(self.__dict__)

    def __repr__(self):
        """Define string representation of ``Edge``.

        Format:
            {
                'source_arn': 'string',
                'destination_arn': 'string',
                'association_type': 'string'
            }

        """
        return "\n\t" + str(self.__dict__)


class Vertex:
    """A vertex for a lineage graph."""

    def __init__(
        self,
        arn: str,
        lineage_entity: str,
        lineage_source: str,
        sagemaker_session,
    ):
        """Initialize ``Vertex`` instance."""
        self.arn = arn
        self.lineage_entity = lineage_entity
        self.lineage_source = lineage_source
        self._session = sagemaker_session

    def __hash__(self):
        """Define hash function for ``Vertex``."""
        return hash(
            (
                "arn",
                self.arn,
                "lineage_entity",
                self.lineage_entity,
                "lineage_source",
                self.lineage_source,
            )
        )

    def __eq__(self, other):
        """Define equal function for ``Vertex``."""
        return (
            self.arn == other.arn
            and self.lineage_entity == other.lineage_entity
            and self.lineage_source == other.lineage_source
        )

    def __str__(self):
        """Define string representation of ``Vertex``.

        Format:
            {
                'arn': 'string',
                'lineage_entity': 'string',
                'lineage_source': 'string',
                '_session': <sagemaker.session.Session object>
            }

        """
        return str(self.__dict__)

    def __repr__(self):
        """Define string representation of ``Vertex``.

        Format:
            {
                'arn': 'string',
                'lineage_entity': 'string',
                'lineage_source': 'string',
                '_session': <sagemaker.session.Session object>
            }

        """
        return "\n\t" + str(self.__dict__)

    def to_lineage_object(self):
        """Convert the ``Vertex`` object to its corresponding lineage object.

        Returns:
            A ``Vertex`` object to its corresponding ``Artifact``,``Action``, ``Context``
            or ``TrialComponent`` object.
        """
        from sagemaker.lineage.context import Context, EndpointContext
        from sagemaker.lineage.action import Action
        from sagemaker.lineage.lineage_trial_component import LineageTrialComponent

        if self.lineage_entity == LineageEntityEnum.CONTEXT.value:
            resource_name = get_resource_name_from_arn(self.arn)
            if self.lineage_source == LineageSourceEnum.ENDPOINT.value:
                return EndpointContext.load(
                    context_name=resource_name, sagemaker_session=self._session
                )
            return Context.load(context_name=resource_name, sagemaker_session=self._session)

        if self.lineage_entity == LineageEntityEnum.ARTIFACT.value:
            return self._artifact_to_lineage_object()

        if self.lineage_entity == LineageEntityEnum.ACTION.value:
            return Action.load(action_name=self.arn.split("/")[1], sagemaker_session=self._session)

        if self.lineage_entity == LineageEntityEnum.TRIAL_COMPONENT.value:
            trial_component_name = get_resource_name_from_arn(self.arn)
            return LineageTrialComponent.load(
                trial_component_name=trial_component_name, sagemaker_session=self._session
            )
        raise ValueError("Vertex cannot be converted to a lineage object.")

    def _artifact_to_lineage_object(self):
        """Convert the ``Vertex`` object to its corresponding ``Artifact``."""
        from sagemaker.lineage.artifact import Artifact, ModelArtifact, ImageArtifact
        from sagemaker.lineage.artifact import DatasetArtifact

        if self.lineage_source == LineageSourceEnum.MODEL.value:
            return ModelArtifact.load(artifact_arn=self.arn, sagemaker_session=self._session)
        if self.lineage_source == LineageSourceEnum.DATASET.value:
            return DatasetArtifact.load(artifact_arn=self.arn, sagemaker_session=self._session)
        if self.lineage_source == LineageSourceEnum.IMAGE.value:
            return ImageArtifact.load(artifact_arn=self.arn, sagemaker_session=self._session)
        return Artifact.load(artifact_arn=self.arn, sagemaker_session=self._session)


class PyvisVisualizer(object):
    """Create object used for visualizing graph using Pyvis library."""

    def __init__(self, graph_styles, pyvis_options: Optional[Dict[str, Any]] = None):
        """Init for PyvisVisualizer.

        Args:
            graph_styles: A dictionary that contains graph style for node and edges by their type.
                Example: Display the nodes with different color by their lineage entity / different
                    shape by start arn.
                        lineage_graph_styles = {
                            "TrialComponent": {
                                "name": "Trial Component",
                                "style": {"background-color": "#f6cf61"},
                                "isShape": "False",
                            },
                            "Context": {
                                "name": "Context",
                                "style": {"background-color": "#ff9900"},
                                "isShape": "False",
                            },
                            "StartArn": {
                                "name": "StartArn",
                                "style": {"shape": "star"},
                                "isShape": "True",
                                "symbol": "★", # shape symbol for legend
                            },
                        }
            pyvis_options(optional): A dict containing PyVis options to customize visualization.
                (see https://visjs.github.io/vis-network/docs/network/#options for supported fields)
        """
        # import visualization packages
        (
            self.Network,
            self.Options,
            self.IFrame,
            self.BeautifulSoup,
        ) = self._import_visual_modules()

        self.graph_styles = graph_styles

        if pyvis_options is None:
            # default pyvis graph options
            pyvis_options = {
                "configure": {"enabled": False},
                "layout": {
                    "hierarchical": {
                        "enabled": True,
                        "blockShifting": True,
                        "direction": "LR",
                        "sortMethod": "directed",
                        "shakeTowards": "leaves",
                    }
                },
                "interaction": {"multiselect": True, "navigationButtons": True},
                "physics": {
                    "enabled": False,
                    "hierarchicalRepulsion": {"centralGravity": 0, "avoidOverlap": None},
                    "minVelocity": 0.75,
                    "solver": "hierarchicalRepulsion",
                },
            }
        # A string representation of a Javascript-like object used to override pyvis options
        self._pyvis_options = f"var options = {dumps(pyvis_options)}"

    def _import_visual_modules(self):
        """Import modules needed for visualization."""
        get_module("pyvis")
        from pyvis.network import Network
        from pyvis.options import Options
        from IPython.display import IFrame

        get_module("bs4")
        from bs4 import BeautifulSoup

        return Network, Options, IFrame, BeautifulSoup

    def _node_color(self, entity):
        """Return node color by background-color specified in graph styles."""
        return self.graph_styles[entity]["style"]["background-color"]

    def _get_legend_line(self, component_name):
        """Generate lengend div line for each graph component in graph_styles."""
        if self.graph_styles[component_name]["isShape"] == "False":
            return '<div><div style="background-color: {color}; width: 1.6vw; height: 1.6vw;\
                display: inline-block; font-size: 1.5vw; vertical-align: -0.2em;"></div>\
                <div style="width: 0.3vw; height: 1.5vw; display: inline-block;"></div>\
                <div style="display: inline-block; font-size: 1.5vw;">{name}</div></div>'.format(
                color=self.graph_styles[component_name]["style"]["background-color"],
                name=self.graph_styles[component_name]["name"],
            )

        return '<div style="background-color: #ffffff; width: 1.6vw; height: 1.6vw;\
            display: inline-block; font-size: 0.9vw; vertical-align: -0.2em;">{shape}</div>\
            <div style="width: 0.3vw; height: 1.5vw; display: inline-block;"></div>\
            <div style="display: inline-block; font-size: 1.5vw;">{name}</div></div>'.format(
            shape=self.graph_styles[component_name]["style"]["shape"],
            name=self.graph_styles[component_name]["name"],
        )

    def _add_legend(self, path):
        """Embed legend to html file generated by pyvis."""
        f = open(path, "r")
        content = self.BeautifulSoup(f, "html.parser")

        legend = """
            <div style="display: inline-block; font-size: 1vw; font-family: verdana;
                vertical-align: top; padding: 1vw;">
        """
        # iterate through graph styles to get legend
        for component in self.graph_styles.keys():
            legend += self._get_legend_line(component_name=component)

        legend += "</div>"

        legend_div = self.BeautifulSoup(legend, "html.parser")

        content.div.insert_after(legend_div)

        html = content.prettify()

        with open(path, "w", encoding="utf8") as file:
            file.write(html)

    def render(self, elements, path="lineage_graph_pyvis.html"):
        """Render graph for lineage query result.

        Args:
            elements: A dictionary that contains the node and the edges of the graph.
                Example:
                    elements["nodes"] contains list of tuples, each tuple represents a node
                        format: (node arn, node lineage source, node lineage entity,
                            node is start arn)
                    elements["edges"] contains list of tuples, each tuple represents an edge
                        format: (edge source arn, edge destination arn, edge association type)

            path(optional): The path/filename of the rendered graph html file.
                (default path: "lineage_graph_pyvis.html")

        Returns:
            display graph: The interactive visualization is presented as a static HTML file.

        """
        net = self.Network(height="600px", width="82%", notebook=True, directed=True)
        net.set_options(self._pyvis_options)

        # add nodes to graph
        for arn, source, entity, is_start_arn in elements["nodes"]:
            entity_text = sub(r"(\w)([A-Z])", r"\1 \2", entity)
            source = sub(r"(\w)([A-Z])", r"\1 \2", source)
            account_id = search(r":\d{12}:", arn)
            name = search(r"\/.*", arn)
            node_info = (
                "Entity: "
                + entity_text
                + "\nType: "
                + source
                + "\nAccount ID: "
                + str(account_id.group()[1:-1])
                + "\nName: "
                + str(name.group()[1:])
            )
            if is_start_arn:  # startarn
                net.add_node(
                    arn,
                    label=source,
                    title=node_info,
                    color=self._node_color(entity),
                    shape="star",
                    borderWidth=3,
                )
            else:
                net.add_node(
                    arn,
                    label=source,
                    title=node_info,
                    color=self._node_color(entity),
                    borderWidth=3,
                )

        # add edges to graph
        for src, dest, asso_type in elements["edges"]:
            net.add_edge(src, dest, title=asso_type, width=2)

        net.write_html(path)
        self._add_legend(path)

        return self.IFrame(path, width="100%", height="600px")


class LineageQueryResult(object):
    """A wrapper around the results of a lineage query."""

    def __init__(
        self,
        edges: List[Edge] = None,
        vertices: List[Vertex] = None,
        startarn: List[str] = None,
    ):
        """Init for LineageQueryResult.

        Args:
            edges (List[Edge]): The edges of the query result.
            vertices (List[Vertex]): The vertices of the query result.
        """
        self.edges = []
        self.vertices = []
        self.startarn = []

        if edges is not None:
            self.edges = edges

        if vertices is not None:
            self.vertices = vertices

        if startarn is not None:
            self.startarn = startarn

    def __str__(self):
        """Define string representation of ``LineageQueryResult``.

        Format:
        {
            'edges':[
                {
                    'source_arn': 'string',
                    'destination_arn': 'string',
                    'association_type': 'string'
                },
            ],

            'vertices':[
                {
                    'arn': 'string',
                    'lineage_entity': 'string',
                    'lineage_source': 'string',
                    '_session': <sagemaker.session.Session object>
                },
            ],

            'startarn':['string', ...]
        }

        """
        return (
            "{"
            + "\n\n".join("'{}': {},".format(key, val) for key, val in self.__dict__.items())
            + "\n}"
        )

    def _covert_edges_to_tuples(self):
        """Convert edges to tuple format for visualizer."""
        edges = []
        # get edge info in the form of (source, target, label)
        for edge in self.edges:
            edges.append((edge.source_arn, edge.destination_arn, edge.association_type))
        return edges

    def _covert_vertices_to_tuples(self):
        """Convert vertices to tuple format for visualizer."""
        verts = []
        # get vertex info in the form of (id, label, class)
        for vert in self.vertices:
            if vert.arn in self.startarn:
                # add "startarn" class to node if arn is a startarn
                verts.append((vert.arn, vert.lineage_source, vert.lineage_entity, True))
            else:
                verts.append((vert.arn, vert.lineage_source, vert.lineage_entity, False))
        return verts

    def _get_visualization_elements(self):
        """Get elements(nodes+edges) for visualization."""
        verts = self._covert_vertices_to_tuples()
        edges = self._covert_edges_to_tuples()

        elements = {"nodes": verts, "edges": edges}
        return elements

    def visualize(
        self,
        path: Optional[str] = "lineage_graph_pyvis.html",
        pyvis_options: Optional[Dict[str, Any]] = None,
    ):
        """Visualize lineage query result.

        Creates a PyvisVisualizer object to render network graph with Pyvis library.
        Pyvis library should be installed before using this method (run "pip install pyvis")
        The elements(nodes & edges) are preprocessed in this method and sent to
        PyvisVisualizer for rendering graph.

        Args:
            path(optional): The path/filename of the rendered graph html file.
                (default path: "lineage_graph_pyvis.html")
            pyvis_options(optional): A dict containing PyVis options to customize visualization.
                (see https://visjs.github.io/vis-network/docs/network/#options for supported fields)

        Returns:
            display graph: The interactive visualization is presented as a static HTML file.
        """
        lineage_graph_styles = {
            # nodes can have shape / color
            "TrialComponent": {
                "name": "Trial Component",
                "style": {"background-color": "#f6cf61"},
                "isShape": "False",
            },
            "Context": {
                "name": "Context",
                "style": {"background-color": "#ff9900"},
                "isShape": "False",
            },
            "Action": {
                "name": "Action",
                "style": {"background-color": "#88c396"},
                "isShape": "False",
            },
            "Artifact": {
                "name": "Artifact",
                "style": {"background-color": "#146eb4"},
                "isShape": "False",
            },
            "StartArn": {
                "name": "StartArn",
                "style": {"shape": "star"},
                "isShape": "True",
                "symbol": "★",  # shape symbol for legend
            },
        }

        pyvis_vis = PyvisVisualizer(lineage_graph_styles, pyvis_options)
        elements = self._get_visualization_elements()
        return pyvis_vis.render(elements=elements, path=path)


class LineageFilter(object):
    """A filter used in a lineage query."""

    def __init__(
        self,
        entities: Optional[List[Union[LineageEntityEnum, str]]] = None,
        sources: Optional[List[Union[LineageSourceEnum, str]]] = None,
        created_before: Optional[datetime] = None,
        created_after: Optional[datetime] = None,
        modified_before: Optional[datetime] = None,
        modified_after: Optional[datetime] = None,
        properties: Optional[Dict[str, str]] = None,
    ):
        """Initialize ``LineageFilter`` instance."""
        self.entities = entities
        self.sources = sources
        self.created_before = created_before
        self.created_after = created_after
        self.modified_before = modified_before
        self.modified_after = modified_after
        self.properties = properties

    def _to_request_dict(self):
        """Convert the lineage filter to its API representation."""
        filter_request = {}
        if self.sources:
            filter_request["Types"] = list(
                map(lambda x: x.value if isinstance(x, LineageSourceEnum) else x, self.sources)
            )
        if self.entities:
            filter_request["LineageTypes"] = list(
                map(lambda x: x.value if isinstance(x, LineageEntityEnum) else x, self.entities)
            )
        if self.created_before:
            filter_request["CreatedBefore"] = self.created_before
        if self.created_after:
            filter_request["CreatedAfter"] = self.created_after
        if self.modified_before:
            filter_request["ModifiedBefore"] = self.modified_before
        if self.modified_after:
            filter_request["ModifiedAfter"] = self.modified_after
        if self.properties:
            filter_request["Properties"] = self.properties
        return filter_request


class LineageQuery(object):
    """Creates an object used for performing lineage queries."""

    def __init__(self, sagemaker_session):
        """Initialize ``LineageQuery`` instance."""
        self._session = sagemaker_session

    def _get_edge(self, edge):
        """Convert lineage query API response to an Edge."""
        return Edge(
            source_arn=edge["SourceArn"],
            destination_arn=edge["DestinationArn"],
            association_type=edge["AssociationType"] if "AssociationType" in edge else None,
        )

    def _get_vertex(self, vertex):
        """Convert lineage query API response to a Vertex."""
        vertex_type = None
        if "Type" in vertex:
            vertex_type = vertex["Type"]
        return Vertex(
            arn=vertex["Arn"],
            lineage_source=vertex_type,
            lineage_entity=vertex["LineageType"],
            sagemaker_session=self._session,
        )

    def _convert_api_response(self, response, converted) -> LineageQueryResult:
        """Convert the lineage query API response to its Python representation."""
        converted.edges = [self._get_edge(edge) for edge in response["Edges"]]
        converted.vertices = [self._get_vertex(vertex) for vertex in response["Vertices"]]

        edge_set = set()
        for edge in converted.edges:
            if edge in edge_set:
                converted.edges.remove(edge)
            edge_set.add(edge)

        vertex_set = set()
        for vertex in converted.vertices:
            if vertex in vertex_set:
                converted.vertices.remove(vertex)
            vertex_set.add(vertex)

        return converted

    def _collapse_cross_account_artifacts(self, query_response):
        """Collapse the duplicate vertices and edges for cross-account."""
        for edge in query_response.edges:
            if (
                "artifact" in edge.source_arn
                and "artifact" in edge.destination_arn
                and edge.source_arn.split("/")[1] == edge.destination_arn.split("/")[1]
                and edge.source_arn != edge.destination_arn
            ):
                edge_source_arn = edge.source_arn
                edge_destination_arn = edge.destination_arn
                self._update_cross_account_edge(
                    edges=query_response.edges,
                    arn=edge_source_arn,
                    duplicate_arn=edge_destination_arn,
                )
                self._update_cross_account_vertex(
                    query_response=query_response, duplicate_arn=edge_destination_arn
                )

        # remove the duplicate edges from cross account
        new_edge = [e for e in query_response.edges if not e.source_arn == e.destination_arn]
        query_response.edges = new_edge

        return query_response

    def _update_cross_account_edge(self, edges, arn, duplicate_arn):
        """Replace the duplicate arn with arn in edges list."""
        for idx, e in enumerate(edges):
            if e.destination_arn == duplicate_arn:
                edges[idx].destination_arn = arn
            elif e.source_arn == duplicate_arn:
                edges[idx].source_arn = arn

    def _update_cross_account_vertex(self, query_response, duplicate_arn):
        """Remove the vertex with duplicate arn in the vertices list."""
        query_response.vertices = [v for v in query_response.vertices if not v.arn == duplicate_arn]

    def query(
        self,
        start_arns: List[str],
        direction: LineageQueryDirectionEnum = LineageQueryDirectionEnum.BOTH,
        include_edges: bool = True,
        query_filter: LineageFilter = None,
        max_depth: int = 10,
    ) -> LineageQueryResult:
        """Perform a lineage query.

        Args:
            start_arns (List[str]): A list of ARNs that will be used as the starting point
                for the query.
            direction (LineageQueryDirectionEnum, optional): The direction of the query.
            include_edges (bool, optional): If true, return edges in addition to vertices.
            query_filter (LineageQueryFilter, optional): The query filter.

        Returns:
            LineageQueryResult: The lineage query result.
        """
        query_response = self._session.sagemaker_client.query_lineage(
            StartArns=start_arns,
            Direction=direction.value,
            IncludeEdges=include_edges,
            Filters=query_filter._to_request_dict() if query_filter else {},
            MaxDepth=max_depth,
        )
        # create query result for startarn info
        query_result = LineageQueryResult(startarn=start_arns)
        query_response = self._convert_api_response(query_response, query_result)
        query_response = self._collapse_cross_account_artifacts(query_response)

        return query_response
