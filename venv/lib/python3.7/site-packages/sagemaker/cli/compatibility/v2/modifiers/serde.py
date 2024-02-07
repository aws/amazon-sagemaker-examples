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
"""Classes to modify SerDe code to be compatibile with version 2.0 and later."""
from __future__ import absolute_import

import ast

from sagemaker.cli.compatibility.v2.modifiers import matching
from sagemaker.cli.compatibility.v2.modifiers.modifier import Modifier

OLD_AMAZON_CLASS_NAMES = {"numpy_to_record_serializer", "record_deserializer"}
NEW_AMAZON_CLASS_NAMES = {"RecordSerializer", "RecordDeserializer"}
OLD_PREDICTOR_CLASS_NAMES = {
    "_CsvSerializer",
    "_JsonSerializer",
    "_NpySerializer",
    "_CsvDeserializer",
    "BytesDeserializer",
    "StringDeserializer",
    "StreamDeserializer",
    "_NumpyDeserializer",
    "_JsonDeserializer",
}

# The values are tuples so that the object can be passed to matching.matches_any.
OLD_CLASS_NAME_TO_NAMESPACES = {
    class_name: ("sagemaker.predictor",) for class_name in OLD_PREDICTOR_CLASS_NAMES
}
OLD_CLASS_NAME_TO_NAMESPACES.update(
    {class_name: ("sagemaker.amazon.common",) for class_name in OLD_AMAZON_CLASS_NAMES}
)

# The values are tuples so that the object can be passed to matching.matches_any.
NEW_CLASS_NAME_TO_NAMESPACES = {
    "CSVSerializer": ("sagemaker.serializers",),
    "JSONSerializer": ("sagemaker.serializers",),
    "NumpySerializer": ("sagemaker.serializers",),
    "CSVDeserializer": ("sagemaker.deserializers",),
    "BytesDeserializer": ("sagemaker.deserializers",),
    "StringDeserializer": ("sagemaker.deserializers",),
    "StreamDeserializer": ("sagemaker.deserializers",),
    "NumpyDeserializer": ("sagemaker.deserializers",),
    "JSONDeserializer": ("sagemaker.deserializers",),
    "RecordSerializer ": ("sagemaker.amazon.common",),
    "RecordDeserializer": ("sagemaker.amazon.common",),
}

OLD_CLASS_NAME_TO_NEW_CLASS_NAME = {
    "_CsvSerializer": "CSVSerializer",
    "_JsonSerializer": "JSONSerializer",
    "_NpySerializer": "NumpySerializer",
    "_CsvDeserializer": "CSVDeserializer",
    "BytesDeserializer": "BytesDeserializer",
    "StringDeserializer": "StringDeserializer",
    "StreamDeserializer": "StreamDeserializer",
    "_NumpyDeserializer": "NumpyDeserializer",
    "_JsonDeserializer": "JSONDeserializer",
    "numpy_to_record_serializer": "RecordSerializer",
    "record_deserializer": "RecordDeserializer",
}

OLD_OBJECT_NAME_TO_NEW_CLASS_NAME = {
    "csv_serializer": "CSVSerializer",
    "json_serializer": "JSONSerializer",
    "npy_serializer": "NumpySerializer",
    "csv_deserializer": "CSVDeserializer",
    "json_deserializer": "JSONDeserializer",
    "numpy_deserializer": "NumpyDeserializer",
}

NEW_CLASS_NAMES = set(OLD_CLASS_NAME_TO_NEW_CLASS_NAME.values())
OLD_CLASS_NAMES = set(OLD_CLASS_NAME_TO_NEW_CLASS_NAME.keys())

OLD_OBJECT_NAMES = set(OLD_OBJECT_NAME_TO_NEW_CLASS_NAME.keys())


class SerdeConstructorRenamer(Modifier):
    """A class to rename SerDe classes."""

    def node_should_be_modified(self, node):
        """Checks if the ``ast.Call`` node instantiates a SerDe class.

        This looks for the following calls (both with and without namespaces):

        - ``sagemaker.predictor._CsvSerializer``
        - ``sagemaker.predictor._JsonSerializer``
        - ``sagemaker.predictor._NpySerializer``
        - ``sagemaker.predictor._CsvDeserializer``
        - ``sagemaker.predictor.BytesDeserializer``
        - ``sagemaker.predictor.StringDeserializer``
        - ``sagemaker.predictor.StreamDeserializer``
        - ``sagemaker.predictor._NumpyDeserializer``
        - ``sagemaker.predictor._JsonDeserializer``
        - ``sagemaker.amazon.common.numpy_to_record_serializer``
        - ``sagemaker.amazon.common.record_deserializer``

        Args:
            node (ast.Call): a node that represents a function call. For more,
                see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the ``ast.Call`` instantiates a SerDe class.
        """
        return matching.matches_any(node, OLD_CLASS_NAME_TO_NAMESPACES)

    def modify_node(self, node):
        """Updates the name and namespace of the ``ast.Call`` node, as applicable.

        This method modifies the ``ast.Call`` node to use the SerDe classes
        available in version 2.0 and later of the Python SDK:

        - ``sagemaker.serializers.CSVSerializer``
        - ``sagemaker.serializers.JSONSerializer``
        - ``sagemaker.serializers.NumpySerializer``
        - ``sagemaker.deserializers.CSVDeserializer``
        - ``sagemaker.deserializers.BytesDeserializer``
        - ``sagemaker.deserializers.StringDeserializer``
        - ``sagemaker.deserializers.StreamDeserializer``
        - ``sagemaker.deserializers.NumpyDeserializer``
        - ``sagemaker.deserializers._JsonDeserializer``
        - ``sagemaker.amazon.common.RecordSerializer``
        - ``sagemaker.amazon.common.RecordDeserializer``

        Args:
            node (ast.Call): a node that represents a SerDe constructor.

        Returns:
            ast.Call: a node that represents the instantiation of a SerDe object.
        """
        class_name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        new_class_name = OLD_CLASS_NAME_TO_NEW_CLASS_NAME[class_name]

        # We don't change the namespace for Amazon SerDe.
        if class_name in OLD_AMAZON_CLASS_NAMES:
            if isinstance(node.func, ast.Name):
                node.func.id = new_class_name
            elif isinstance(node.func, ast.Attribute):
                node.func.attr = new_class_name
            return node

        namespace_name = NEW_CLASS_NAME_TO_NAMESPACES[new_class_name][0]
        subpackage_name = namespace_name.split(".")[1]
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id=subpackage_name), attr=new_class_name),
            args=[],
            keywords=[],
        )


class SerdeKeywordRemover(Modifier):
    """A class to remove Serde-related keyword arguments from call expressions."""

    def node_should_be_modified(self, node):
        """Checks if the ``ast.Call`` node uses deprecated keywords.

        In particular, this function checks if:

        - The ``ast.Call`` represents the ``create_model`` method.
        - Either the serializer or deserializer keywords are used.

        Args:
            node (ast.Call): a node that represents a function call. For more,
                see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the ``ast.Call`` contains keywords that should be removed.
        """
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "create_model":
            return False
        return any(keyword.arg in {"serializer", "deserializer"} for keyword in node.keywords)

    def modify_node(self, node):
        """Removes the serializer and deserializer keywords, as applicable.

        Args:
            node (ast.Call): a node that represents a ``create_model`` call.

        Returns:
            ast.Call: the node that represents a ``create_model`` call without
                serializer or deserializers keywords.
        """
        i = 0
        while i < len(node.keywords):
            keyword = node.keywords[i]
            if keyword.arg in {"serializer", "deserializer"}:
                node.keywords.pop(i)
            else:
                i += 1
        return node


class SerdeObjectRenamer(Modifier):
    """A class to rename SerDe objects imported from ``sagemaker.predictor``."""

    def node_should_be_modified(self, node):
        """Checks if the ``ast.Name`` node identifies a SerDe object.

        This looks for the following objects:

        - ``sagemaker.predictor.csv_serializer``
        - ``sagemaker.predictor.json_serializer``
        - ``sagemaker.predictor.npy_serializer``
        - ``sagemaker.predictor.csv_deserializer``
        - ``sagemaker.predictor.json_deserializer``
        - ``sagemaker.predictor.numpy_deserializer``

        Args:
            node (ast.Call): a node that represents a function call. For more,
                see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the ``ast.Call`` instantiates a SerDe class.
        """
        name = node.id if isinstance(node, ast.Name) else node.attr
        return name in OLD_OBJECT_NAMES

    def modify_node(self, node):
        """Replaces the ``ast.Name`` node with a ``ast.Call`` node.

        Replaced node instantiates a class available in version 2.0 and later of the Python SDK.

        - ``sagemaker.serializers.CSVSerializer()``
        - ``sagemaker.serializers.JSONSerializer()``
        - ``sagemaker.serializers.NumpySerializer()``
        - ``sagemaker.deserializers.CSVDeserializer()``
        - ``sagemaker.deserializers.JSONDeserializer()``
        - ``sagemaker.deserializers.NumpyDeserializer()``

        The ``sagemaker`` prefix is excluded from the returned node.

        Args:
            node (ast.Name): a node that represents a Python identifier.

        Returns:
            ast.Call: a node that represents the instantiation of a SerDe object.
        """
        object_name = node.id if isinstance(node, ast.Name) else node.attr
        new_class_name = OLD_OBJECT_NAME_TO_NEW_CLASS_NAME[object_name]
        namespace_name = NEW_CLASS_NAME_TO_NAMESPACES[new_class_name][0]
        subpackage_name = namespace_name.split(".")[1]
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id=subpackage_name), attr=new_class_name),
            args=[],
            keywords=[],
        )


class SerdeImportFromPredictorRenamer(Modifier):
    """A class to update import statements starting with ``from sagemaker.predictor``."""

    def node_should_be_modified(self, node):
        """Checks if the import statement imports a SerDe from the ``sagemaker.predictor`` module.

        Args:
            node (ast.ImportFrom): a node that represents a ``from ... import ... `` statement.
                For more, see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: True if and only if the ``ast.ImportFrom`` imports a SerDe
                from the ``sagemaker.predictor`` module.
        """
        return node.module == "sagemaker.predictor" and any(
            name.name in (OLD_CLASS_NAMES | OLD_OBJECT_NAMES) for name in node.names
        )

    def modify_node(self, node):
        """Removes the imported SerDe classes, as applicable.

        Args:
            node (ast.ImportFrom): a node that represents a ``from ... import ... `` statement.
                For more, see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            ast.ImportFrom: a node that represents a import statement, which has
                been modified to remove imported serializers. If nothing is
                imported, None is returned.
        """
        i = 0
        while i < len(node.names):
            name = node.names[i].name
            if name in OLD_CLASS_NAMES | OLD_OBJECT_NAMES:
                node.names.pop(i)
            else:
                i += 1

        return node if node.names else None


class SerdeImportFromAmazonCommonRenamer(Modifier):
    """A class to update import statements starting with ``from sagemaker.amazon.common``."""

    def node_should_be_modified(self, node):
        """Checks if the import statement imports a SerDe from the ``sagemaker.amazon.common``.

        This checks for:
        - ``sagemaker.amazon.common.numpy_to_record_serializer``
        - ``sagemaker.amazon.common.record_deserializer``

        Args:
            node (ast.ImportFrom): a node that represents a ``from ... import ... `` statement.
                For more, see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: True if and only if the ``ast.ImportFrom`` imports a SerDe from
                the ``sagemaker.amazon.common`` module.
        """
        return node.module == "sagemaker.amazon.common" and any(
            alias.name in OLD_AMAZON_CLASS_NAMES for alias in node.names
        )

    def modify_node(self, node):
        """Upgrades the ``numpy_to_record_serializer`` and ``record_deserializer`` imports.

        This upgrades the classes to (if applicable):
        - ``sagemaker.amazon.common.RecordSerializer``
        - ``sagemaker.amazon.common.RecordDeserializer``

        Args:
            node (ast.ImportFrom): a node that represents a ``from ... import ... `` statement.
                For more, see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            ast.ImportFrom: a node that represents a import statement, which has
                been modified to import the upgraded class name.
        """
        for alias in node.names:
            if alias.name in OLD_AMAZON_CLASS_NAMES:
                alias.name = OLD_CLASS_NAME_TO_NEW_CLASS_NAME[alias.name]
        return node


class _ImportInserter(Modifier):
    """A class to insert import statements into the Python module."""

    def __init__(self, class_names, import_node):
        """Initialize the ``class_names`` and ``import_node`` attributes.

        Args:
            class_names (set): If any of these class names are referenced in the
                module, then ``import_node`` is inserted.
            import_node (ast.ImportFrom): The AST node to insert.
        """
        self.class_names = class_names
        self.import_node = import_node

    def node_should_be_modified(self, module):
        """Checks if the ``ast.Module`` node contains references to the specified class names.

        Args:
            node (ast.Module): a node that represents a Python module. For more,
                see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the ``ast.Module`` references one of the specified classes.
        """
        for node in ast.walk(module):
            if isinstance(node, ast.Name) and node.id in self.class_names:
                return True
            if isinstance(node, ast.Attribute) and node.attr in self.class_names:
                return True
        return False

    def modify_node(self, module):
        """Modifies the ``ast.Module`` node by inserted the specified node.

        The ``import_node`` object is inserted immediately before the first
        import statement.

        Args:
            node (ast.Module): a node that represents a Python module.

        Returns:
            ast.Module: a node that represents a Python module, which has been
                modified to import a module.
        """
        for i, node in enumerate(module.body):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module.body.insert(i, self.import_node)
                return module

        module.body.insert(0, self.import_node)
        return module


class SerializerImportInserter(_ImportInserter):
    """A class to import the ``sagemaker.serializers`` module, if necessary.

    This looks for references to the following classes:

    - ``sagemaker.serializers.CSVSerializer``
    - ``sagemaker.serializers.JSONSerializer``
    - ``sagemaker.serializer.NumpySerializer``

    Because ``SerializerImportInserter`` is gauranteed to run after
    ``SerdeConstructorRenamer`` and ``SerdeObjectRenamer``,
    we only need to check for the new serializer class names.
    """

    def __init__(self):
        """Initialize the ``class_names`` and ``import_node`` attributes.

        Amazon-specific serializers are ignored because they are not defined in
        the ``sagemaker.serializers`` module.
        """
        class_names = {
            class_name
            for class_name in NEW_CLASS_NAMES - NEW_AMAZON_CLASS_NAMES
            if "Serializer" in class_name
        }
        import_node = ast.ImportFrom(
            module="sagemaker", names=[ast.alias(name="serializers", asname=None)], level=0
        )
        super().__init__(class_names, import_node)


class DeserializerImportInserter(_ImportInserter):
    """A class to import the ``sagemaker.deserializers`` module, if necessary.

    This looks for references to the following classes:

    - ``sagemaker.serializers.CSVDeserializer``
    - ``sagemaker.serializers.BytesDeserializer``
    - ``sagemaker.serializers.StringDeserializer``
    - ``sagemaker.serializers.StreamDeserializer``
    - ``sagemaker.serializers.NumpyDeserializer``
    - ``sagemaker.serializer.JSONDeserializer``

    Because ``DeserializerImportInserter`` is gauranteed to run after
    ``SerdeConstructorRenamer`` and ``SerdeObjectRenamer``,
    we only need to check for the new deserializer class names.
    """

    def __init__(self):
        """Initialize the ``class_names`` and ``import_node`` attributes.

        Amazon-specific deserializers are ignored because they are not defined
        in the ``sagemaker.deserializers`` module.
        """
        class_names = {
            class_name
            for class_name in NEW_CLASS_NAMES - NEW_AMAZON_CLASS_NAMES
            if "Deserializer" in class_name
        }
        import_node = ast.ImportFrom(
            module="sagemaker", names=[ast.alias(name="deserializers", asname=None)], level=0
        )
        super().__init__(class_names, import_node)
