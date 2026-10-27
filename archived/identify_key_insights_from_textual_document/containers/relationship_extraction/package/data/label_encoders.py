import json
from pathlib import Path
from typing import Dict
from typing import List
from typing import Union


class LabelEncoder:
    def __init__(self, str_to_id_mapping: Dict[str, int]) -> None:
        """
        Converts back and forth between human readable string labels and
        their associated integer ids used by loss function.

        Args:
            str_to_id_mapping (Dict[str, int]):
                mapping from string label to integer ids
        """
        self._str_to_id_mapping = str_to_id_mapping
        self._id_to_str_mapping = {i: s for s, i in str_to_id_mapping.items()}

    @classmethod
    def from_file(cls, file_path: Union[Path, str]) -> "LabelEncoder":
        """
        Construct a label encoder from a valid json file.

        Args:
            file_path (Union[Path, str]): path to json file
        Returns:
            LabelEncoder: associated label encoder
        """
        with open(file_path, "r") as f:
            str_to_id_mapping = json.load(f)
        return cls(str_to_id_mapping)

    @classmethod
    def from_str_list(cls, strings: List[str]) -> "LabelEncoder":
        """
        Construct a label encoder from a list of string labels by
        automatically generating the integer ids in ascending order.

        Args:
            strings (List[str]): list of string labels.

        Returns:
            LabelEncoder: associated label encoder
        """
        str_to_id_mapping = {s: i for i, s in enumerate(strings)}
        return cls(str_to_id_mapping)

    def save(self, file_path: Union[Path, str]) -> None:
        """
        Saves the label encoder to a file in json format.
        Can be loaded back using `from_file` method.

        Args:
            file_path (Union[Path, str]): path for json file
        """
        with open(file_path, "w") as f:
            json.dump(self._str_to_id_mapping, f)

    def str_to_id(self, string: str) -> int:
        """
        Converts from string label to integer id.

        Args:
            string (str): string label

        Returns:
            int: integer id
        """
        return self._str_to_id_mapping[string]

    def id_to_str(self, id: int) -> str:
        """
        Converts from integer id to string label.

        Args:
            id (int): integer id

        Returns:
            str: string label
        """
        return self._id_to_str_mapping[id]

    def __len__(self) -> int:
        """
        Get number of labels.

        Returns:
            int: number of labels
        """
        return len(self._str_to_id_mapping)

    def __str__(self) -> str:
        output = "{:>6}  {}\n".format("id", "str")
        for i in range(len(self)):
            output += "{:>6}  {}\n".format(i, self.id_to_str(i))
        return output
