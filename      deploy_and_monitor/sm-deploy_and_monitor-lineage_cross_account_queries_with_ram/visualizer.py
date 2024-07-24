from pyvis.network import Network
import os


class Visualizer:
    def __init__(self):
        self.directory = "generated"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def render(self, query_lineage_response, scenario_name):
        net = self.get_network()
        for vertex in query_lineage_response["Vertices"]:
            arn = vertex["Arn"]
            if "Type" in vertex:
                label = vertex["Type"]
            else:
                label = None
            lineage_type = vertex["LineageType"]
            name = self.get_name(arn)
            title = self.get_title(arn)
            net.add_node(vertex["Arn"], label=label+"\n"+lineage_type, title=title, shape="circle")

        for edge in query_lineage_response["Edges"]:
            source = edge["SourceArn"]
            dest = edge["DestinationArn"]
            net.add_edge(dest, source)

        return net.show(f"{self.directory}/{scenario_name}.html")

    def get_title(self, arn):
        return f"Arn: {arn}"

    def get_name(self, arn):
        name = arn.split("/")[1]
        return name

    def get_network(self):
        net = Network(height="400px", width="800px", directed=True, notebook=True)
        net.set_options(
            """
        var options = {
  "nodes": {
    "borderWidth": 3,
    "shadow": {
      "enabled": true
    },
    "shapeProperties": {
      "borderRadius": 3
    },
    "size": 11,
    "shape": "circle"
  },
  "edges": {
    "arrows": {
      "to": {
        "enabled": true
      }
    },
    "color": {
      "inherit": true
    },
    "smooth": false
  },
  "layout": {
    "hierarchical": {
      "enabled": true,
      "direction": "LR",
      "sortMethod": "directed"
    }
  },
  "physics": {
    "hierarchicalRepulsion": {
      "centralGravity": 0
    },
    "minVelocity": 0.75,
    "solver": "hierarchicalRepulsion"
  }
}
        """
        )
        return net
