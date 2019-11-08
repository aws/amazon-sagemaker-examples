import logging


class ConfigurationList(object):
    """Helper Object for converting CLI arguments (or SageMaker hyperparameters)
    into Coach configuration.
    """

    def __init__(self):
        """Args:
            - arg_list [list]: list of arguments on the command-line like [key1, value1, key2, value2, ...]
            - prefix [str]: Prefix for every key that must be present, e.g. "--" for common command-line args
        """
        self.hp_dict = {}

    def store(self, name, value):
        """Store a key/value hyperparameter combination
        """
        self.hp_dict[name] = value

    def apply_subset(self, config_object, prefix):
        """Merges configured hyperparameters in the params dict into the config_object.
        Recognized arguments are consumed out of self.hp_dict

        Args:
            config_object (obj):  will be something like a Coach TaskParameters object, where we're setting properties
            params (dict): comes from the command line (and thus customer-specified hyperparameters)
            prefix (str): string prefix for which items in params to use.  (e.g. "rl.task_params.")
        """
        # Materialize a copy of the dict as tuples so we can modify the original dict as we go.
        for key, val in list(self.hp_dict.items()):
            if key.startswith(prefix):
                logging.debug("Configuring %s with %s=%s" % (prefix, key, val))
                subkey = key[ len(prefix): ]
                msg = "%s%s=%s" % (prefix, subkey, val)
                try:
                    self._set_rl_property_value(config_object, subkey, val, prefix)
                except:
                    print("Failure while applying hyperparameter %s" % msg)
                    raise
                del self.hp_dict[key]

    def _set_rl_property_value(self, obj, key, val, path=""):
        """Sets a property on obj to val, or to a sub-object within obj if key looks like "foo.bar"
        """
        if key.find(".") >= 0:
            top_key, sub_keys = key_list = key.split(".",1)
            if top_key.startswith("__"):
                raise ValueError("Attempting to set unsafe property name %s" % top_key)
            if isinstance(obj,dict):
                sub_obj = obj[top_key]
            else:
                sub_obj = obj.__dict__[top_key]
            # Recurse
            return self._set_rl_property_value(sub_obj, sub_keys, val, "%s.%s" % (path,top_key) )
        else:
            key, val = self._parse_type(key,val)
            if key.startswith("__"):
                raise ValueError("Attempting to set unsafe property name %s" % key)
            if isinstance(obj, dict):
                obj[key] = val
            else:
                obj.__dict__[key] = val

    def _autotype(self, val):
        """Converts string to an int or float as possible.
        """
        if type(val) == dict:
            return val
        if type(val) == list:
            return val
        if type(val) == bool:
            return val
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        return val

    # Being security-paranoid and not instantiating any arbitrary string the customer passes in
    ALLOWED_TYPES = {}

    def _parse_type(self, key, val):
        """Converts the val to an appropriately typed Python object.
        Automatically detects ints and floats when possible.
        If the key takes the form "foo:bar" then it looks in ALLOWED_TYPES
        for an entry of bar, and instantiates one of those objects, passing
        val to the constructor.  So if key="foo:EnvironmentSteps" then
        """
        val = self._autotype(val)
        if key.find(":") > 0:
            key, obj_type = key.split(":", 1)
            cls = self.ALLOWED_TYPES.get(obj_type)
            if not cls:
                raise ValueError("Unrecognized object type %s.  Allowed values are %s" % (obj_type, self.ALLOWED_TYPES.keys()))
            val = cls(val)
        return key, val
