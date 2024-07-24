import json
from pathlib import Path
from typing import List
from typing import Union

from package.objects import Relationship


def to_json_lines(relationships: List[Relationship], file_path: Union[Path, str]) -> None:
    json_lines = []
    for relationship in relationships:
        json_line = relationship.to_json_str()
        json_lines.append(json_line)
    with open(file_path, "w") as f:
        f.writelines("\n".join(json_lines))


def from_json_lines(file_path: Union[Path, str]) -> List[Relationship]:
    with open(file_path, "r") as f:
        lines = f.readlines()
    relationships = []
    for line in lines:
        args_dict = json.loads(line)
        relationship = Relationship.from_dict(args_dict)
        relationships.append(relationship)
    return relationships
