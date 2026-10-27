import os
import socket
import xml.etree.ElementTree as ET


class socket_builder:
    """
    Helper class that creates socket.cfg files for all EnergyPlus instances

    Arguments
            path -- path to main folder where we need to store the socket.cfg file

    """

    def __init__(self, path):
        self.path = path

    def build(self):
        # work in the same directory as the input
        with cd(self.path):
            configs = []
            port = self.get_free_port()
            configs.append(port)
            # write the port configuration to socket.cfg
            xml = self.build_XML(port)
        return configs

    def get_free_port(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
        s.close()
        return port

    def build_XML(self, port):
        # format of the XML is dictacted by EnergyPlus
        tree = ET.ElementTree()
        bcvtb_client = ET.Element("BCVTB-client")
        ipc = ET.SubElement(bcvtb_client, "ipc")
        socket_ele = ET.SubElement(ipc, "socket")
        socket_ele.set("port", str(port))
        socket_ele.set("hostname", "localhost")
        tree._setroot(bcvtb_client)
        tree.write("socket.cfg", encoding="ISO-8859-1")


class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
