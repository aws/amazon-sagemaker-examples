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
"""Classes for updating code in files for version 2.0 or later of the SageMaker Python SDK."""
from __future__ import absolute_import

from abc import abstractmethod
import json
import logging
import os

import pasta

from sagemaker.cli.compatibility.v2.ast_transformer import ASTTransformer

LOGGER = logging.getLogger(__name__)


class FileUpdater(object):
    """An abstract class for updating files."""

    def __init__(self, input_path, output_path):
        """Creates ``FileUpdater`` for updating a file to be compatible with version 2.0 and later.

        Args:
            input_path (str): Location of the input file.
            output_path (str): Desired location for the output file.
                If the directories don't already exist, then they are created.
                If a file exists at ``output_path``, then it is overwritten.
        """
        self.input_path = input_path
        self.output_path = output_path

    @abstractmethod
    def update(self):
        """Reads, updates and writes the code for version 2.0 and later of the SageMaker Python SDK.

        Reads the input file, updates the code so that it is
        compatible with version 2.0 and later of the SageMaker Python SDK,
        and writes the updated code to an output file.
        """

    def _make_output_dirs_if_needed(self):
        """Checks if the directory path for ``self.output_path`` exists, and creates it if not.

        This function also logs a warning if ``self.output_path`` already exists.
        """
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if os.path.exists(self.output_path):
            LOGGER.warning("Overwriting file %s", self.output_path)


class PyFileUpdater(FileUpdater):
    """A class for updating Python (``*.py``) files."""

    def update(self):
        """Reads, updates and writes the code for version 2.0 and later of the SageMaker Python SDK.

        Reads the input Python file, updates the code so that it is
        compatible with version 2.0 and later of the SageMaker Python SDK,
        and writes the updated code to an output file.
        """
        output = self._update_ast(self._read_input_file())
        self._write_output_file(output)

    def _update_ast(self, input_ast):
        """Updates an abstract syntax tree (AST).

        So that it is compatible with version 2.0 and later of the SageMaker Python SDK.

        Args:
            input_ast (ast.Module): AST to be updated for use with
                the Python SDK version 2.0 and later.

        Returns:
            ast.Module: Updated AST that is compatible with the Python SDK version 2.0 and later.
        """
        return ASTTransformer().visit(input_ast)

    def _read_input_file(self):
        """Reads input file and parses it as an abstract syntax tree (AST).

        Returns:
            ast.Module: AST representation of the input file.
        """
        with open(self.input_path) as input_file:
            return pasta.parse(input_file.read())

    def _write_output_file(self, output):
        """Writes abstract syntax tree (AST) to output file.

        Creates the directories for the output path, if needed.

        Args:
            output (ast.Module): AST to save as the output file.
        """
        self._make_output_dirs_if_needed()

        with open(self.output_path, "w") as output_file:
            output_file.write(pasta.dump(output))


class JupyterNotebookFileUpdater(FileUpdater):
    """A class for updating Jupyter notebook (``*.ipynb``) files.

    For more on this file format, see
    https://ipython.org/ipython-doc/dev/notebook/nbformat.html#nbformat.
    """

    def update(self):
        """Reads, updates and writes the code for version 2.0 and later of the SageMaker Python SDK.

        Reads the input Jupyter notebook file, updates the code so that it is
        compatible with version 2.0 and later of the SageMaker Python SDK, and writes the
        updated code to an output file.
        """
        nb_json = self._read_input_file()
        for cell in nb_json["cells"]:
            if cell["cell_type"] == "code" and not self._contains_shell_cmds(cell):
                updated_source = self._update_code_from_cell(cell)
                cell["source"] = updated_source

        self._write_output_file(nb_json)

    def _contains_shell_cmds(self, cell):
        """Checks if the cell's source uses either ``%%`` or ``!`` to execute shell commands.

        Args:
            cell (dict): A dictionary representation of a code cell from
                a Jupyter notebook. For more info, see
                https://ipython.org/ipython-doc/dev/notebook/nbformat.html#code-cells.

        Returns:
            bool: If the first line starts with ``%%`` or any line starts with ``!``.
        """
        source = cell["source"]

        if source[0].startswith("%"):
            return True

        return any(line.startswith("!") for line in source)

    def _update_code_from_cell(self, cell):
        """Updates the code from a code cell.

        So that it is compatible with version 2.0 and later of the SageMaker Python SDK.

        Args:
            cell (dict): A dictionary representation of a code cell from
                a Jupyter notebook. For more info, see
                https://ipython.org/ipython-doc/dev/notebook/nbformat.html#code-cells.

        Returns:
            list[str]: A list of strings containing the lines of updated code that
                can be used for the "source" attribute of a Jupyter notebook code cell.
        """
        code = "".join(cell["source"])
        updated_ast = ASTTransformer().visit(pasta.parse(code))
        updated_code = pasta.dump(updated_ast)
        return self._code_str_to_source_list(updated_code)

    def _code_str_to_source_list(self, code):
        r"""Converts a string of code into a list for a Jupyter notebook code cell.

        Args:
            code (str): Code to be converted.

        Returns:
            list[str]: A list of strings containing the lines of code that
                can be used for the "source" attribute of a Jupyter notebook code cell.
                Each element of the list (i.e. line of code) contains a
                trailing newline character ("\n") except for the last element.
        """
        source_list = ["{}\n".format(s) for s in code.split("\n")]
        source_list[-1] = source_list[-1].rstrip("\n")
        return source_list

    def _read_input_file(self):
        """Reads input file and parses it as JSON.

        Returns:
            dict: JSON representation of the input file.
        """
        with open(self.input_path) as input_file:
            return json.load(input_file)

    def _write_output_file(self, output):
        """Writes JSON to output file. Creates the directories for the output path, if needed.

        Args:
            output (dict): JSON to save as the output file.
        """
        self._make_output_dirs_if_needed()

        with open(self.output_path, "w") as output_file:
            json.dump(output, output_file, indent=1)
            output_file.write("\n")  # json.dump does not write trailing newline
