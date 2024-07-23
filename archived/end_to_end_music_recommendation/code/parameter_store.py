"""
A simple class for loading and storing JSON documents in a local file.
"""

import json
import os.path
import pprint
from datetime import datetime

class ParameterStore():
    """
    Create a parameter store with namespace functionality
    that stores keys and values in a local JSON file.
    """
    def __init__(self, path='', parameters={}, filename='parameters.json', verbose = True):
        """
        Constructor
        """
        self.filename = path + filename
        self.verbose = verbose

        # If the file already exists, load up the params
        if os.path.exists(self.filename):
            self.load()
            self.parameters.update(parameters)
        # Otherwise set params to be empty
        else:
            self.parameters = parameters

    def set_namespace(self, namespace):
        self.namespace = namespace
        
    def create(self, parameters={}, namespace='experiment_1'):
        """
        Create a new parameter store with the option of
        separating keys and values by namespace.

        :param parameters: dict
        :param namespace: str
        :return: None
        """
        self.namespace = namespace
        self.parameters[namespace] = parameters
        if self.verbose:
            print (f"Creating : \n")
            pprint.pprint(parameters)


    def read(self, namespace='experiment_1'):
        """
        Return a dictionary of parameters.

        :param namespace: str
        :return: dict
        """
        self.namespace = namespace
        try:
            if self.parameters:
                if self.verbose:
                    print (f"Reading : {namespace}\n")
                    pprint.pprint(self.parameters)
                return self.parameters[namespace]
            else:
                return None
            

        except Exception as inst:
            print(type(inst))       # the exception instance
            print(inst.args)        # arguments stored in .args
            print(inst) 
            

        
    def render(self, parameters={}, namespace='experiment_1'):
        if self.parameters:
            pprint (self.parameters)

    def add(self, parameters={}, namespace='experiment_1'):
        """
        Add new parameters including updating old ones with new values .

        :param parameters: dict
        :param namespace: str
        :return: None
        """
        try:
            self.parameters[namespace].update(parameters)
            if self.verbose:
                print (f"Updating Params : \n")
                pprint.pprint(parameters)
                

        except Exception as inst:
            print(type(inst))       # the exception instance
            print(inst.args)        # arguments stored in .args
            print(inst)   


    def delete(self, key, namespace='experiment_1'):
        """
        Delete a key and its value from the parameter store.

        :param key: str
        :param namespace: str
        :return: None
        """
    
        self.parameters[namespace].__delitem__(key)


    def clear(self, namespace='experiment_1'):
        """
        Erase all parameter keys and values in a given namespace.

        :param namespace: str
        :return: None
        """
        self.parameters[namespace] = {}


    def clear_all(self):
        """
        Erase all parameter keys and in every namespace.

        :return: None
        """
        self.parameters = {}


    def load(self):
        """
        Load existing parameters from the given file name
        and namespace.

        :return: None
        """
        with open(self.filename, 'r') as file:
            document = file.read()

        self.parameters = json.loads(document)
        if self.verbose:
                print (f"Loading : \n")
                pprint.pprint(self.parameters)


    def store(self):
        """
        Save the updates to the parameter store.

        :return: None
        """
        ordered = dict(sorted(self.parameters.items()))
 
        self.parameters = ordered
   
        # get and update datetime for when you are storing
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print("date and time =", dt_string)
        
        self.add ({'__timestamp': dt_string},namespace = self.namespace)
        if self.verbose:
            print (f"Storing : \n")
            pprint.pprint(self.parameters)
        with open(self.filename, 'w') as file:
            file.write(json.dumps(self.parameters))