import logging
from collections import OrderedDict


class Module(object):
    """Base class for a Module similar to pytorch modules.
    This class will be subclassed by other classes.

    Args:
        name: Supply a name for the module.
    """

    def __init__(self, name):
        self._modules = OrderedDict()
        self._leaf = False
        self._name = name
        self.output = None
        self.description = []

    def get_params(self):
        """Returns the parameters of the module, including all the sub parameters."""
        if not self._leaf:
            params = {}
            for module in self._modules.values():
                params[module.get_name()] = module.get_params()[module.get_name()]
        return params

    def add_module(self, module, name):
        """Adds a child module to the current module.

        Args:
            module: Supply another module here.
            name: A name to register the child with.
        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module Subclass".format(type(module)))
        if self._leaf:
            self._leaf = False
        if name in self._modules.keys():
            raise ValueError("{} already exists in this module, choose another name")
        else:
            self._modules[name] = module
        self.output = module.output

    def get_name(self):
        """Returns the modeule name"""
        return self._name

    def get_desc(self):
        """Retruns the module description"""
        return self.description

    def pretty_print(self, prefix=""):
        """Prints module detail in context with all sub modules"""
        if self._leaf:
            logging.info(prefix + "------------------------------")
            logging.info(prefix + "Module Name: " + self.get_name())
            for desc in self.get_desc():
                logging.info(prefix + " |- Property: " + str(desc))
            logging.info(prefix + "------------------------------")
        else:
            logging.info(prefix + "Module: " + self.get_name())
            logging.info(prefix + "    |")
            for module in self._modules.keys():
                self._modules[module].pretty_print(prefix + "    ")

    def get_sub_module(self, mod_name):
        """Returns the sub module of some name"""
        if not isinstance(mod_name, str):
            mod_name = str(mod_name)
        if mod_name in self._modules.keys():
            return self._modules[mod_name]
        else:
            for mod in self._modules.keys():
                out = self.get_sub_module(mod).get_sub_module(mod_name)
                if not out is None:
                    return out
        return None


class Layer(Module):
    """Base class for writing layers. This class will be subclassed by others
    Args:
        name: Supply a name.
        start: Starting layer number.
        end: Ending layer number.
    """

    def __init__(self, name, start=None, end=None):
        super(Layer, self).__init__(name)
        self._leaf = True
        self.description = [start, end]

    def add_module(self):
        raise TypeError("Layers can't take in more modules.")
