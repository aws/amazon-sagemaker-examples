import re
from collections import OrderedDict
from pathlib import Path
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

from package.objects import Entity
from package.objects import Relationship
from package.objects import Source
from package.objects import Statement


def find_special_tokens(text: str, special_tokens: List[str]) -> OrderedDict:
    """
    Given text and a list of special tokens, this funtion will find the
    character index of each special token. Order is of the special tokens
    is enforced and an AssertionError will be raised if the special tokens
    are found out of order or duplicated.

    Args:
        text (str): text containing the special tokens
        special_tokens (List[str]): a list of special tokens (where order
            is important)

    Returns:
        OrderedDict: character index of each special token. Order same as
            special token input, but can use dictionary lookup.
    """
    idxs = OrderedDict()
    last_idx = None
    last_token = None
    special_chars = 0
    for token in special_tokens:
        idx = text.find(token)
        assert idx != -1, f"{token} not found in text."
        if last_idx:
            assert idx > last_idx, f"{token} found before {last_token}"
        assert text[idx + len(token) :].find(token) == -1, f"{token} duplicated in text."
        idxs[token] = idx - special_chars
        last_idx = idx
        last_token = token
        special_chars += len(token)
    return idxs


def remove_special_tokens(text: str, special_tokens: List[str]) -> str:
    """
    Given text and a list of special tokens, this funtion will return the
    text with the special tokens removed. An AssertionError will be raised
    if the special token is not found, is duplicate or overlaps with
    another special token.

    Args:
        text (str): text containing the special tokens
        special_tokens (List[str]): a list of special tokens

    Returns:
        str: text without the special tokens
    """
    token_idxs: Set[int] = set()
    for token in special_tokens:
        start_idx = text.find(token)
        assert start_idx != -1, f"{token} not found in text."
        end_idx = start_idx + len(token)
        assert text[end_idx:].find(token) == -1, f"{token} duplicated in text."
        idxs = range(start_idx, end_idx)
        overlap = token_idxs.intersection(idxs)
        assert len(overlap) == 0, f"{token} overlaps another special token at {idxs}."
        token_idxs.update(idxs)
    text = "".join([c for i, c in enumerate(text) if i not in token_idxs])
    return text


def parse_file(file_path: Union[Path, str], sample_idx_offset: Optional[int] = None) -> List[Relationship]:
    file_path = Path(file_path)
    with open(file_path, "r") as f:
        lines = f.readlines()
    num_lines = len(lines)
    num_samples = int(num_lines / 4)
    assert num_samples * 4 == num_lines, f"Should have 4 lines per block, but {num_lines} cannot be divided by 4."
    relationships = []
    for sample_idx in range(num_samples):
        block = lines[(sample_idx * 4) : ((sample_idx + 1) * 4)]
        relationship = parse_block(block)
        if sample_idx_offset:
            assert relationship.source.sample_idx == (sample_idx + 1 + sample_idx_offset)
        relationships.append(relationship)
    return relationships


def parse_block(block: List[str]) -> Relationship:
    text, label, comment, blank = block
    entity_one, entity_two, statement, source = parse_text(text)
    label, is_reversed = parse_label(label)
    relationship = Relationship(
        entity_one=entity_one,
        entity_two=entity_two,
        statement=statement,
        source=source,
        label=label,
        label_seperator="-",
        is_reversed=is_reversed,
    )
    return relationship


def parse_text(text: str) -> Tuple[Entity, Entity, Statement, dict]:
    sample_idx_str, text = text.split("\t")
    sample_idx = int(sample_idx_str)
    source = Source(sample_idx=sample_idx)
    text = text.strip('"')
    special_tokens = ["<e1>", "</e1>", "<e2>", "</e2>"]
    idxs = find_special_tokens(text, special_tokens)
    text = remove_special_tokens(text, special_tokens)
    entity_one = Entity(text=text[idxs["<e1>"] : idxs["</e1>"]], start_char=idxs["<e1>"], end_char=idxs["</e1>"])
    entity_two = Entity(text=text[idxs["<e2>"] : idxs["</e2>"]], start_char=idxs["<e2>"], end_char=idxs["</e2>"])
    statement = Statement(text=text, start_char=0, end_char=len(text))
    return entity_one, entity_two, statement, source


def parse_label(label: str) -> Tuple[str, bool]:
    pattern = r"^(.*)\(e([12]),e([12])\)$"
    match = re.search(pattern, label)
    if match:
        label, first, second = match.groups()
        if first == "1" and second == "2":
            is_reversed = False
        elif first == "2" and second == "1":
            is_reversed = True
        else:
            raise ValueError("Cannot infer direction")
    else:
        label = label.rstrip("\n")
        if label == "Other":
            is_reversed = False
        else:
            raise ValueError(f"Cannot find {pattern} in {label}")
    return label, is_reversed


def label_set(file_path: Union[Path, str], sample_idx_offset: Optional[int] = None) -> set:
    relationships = parse_file(file_path, sample_idx_offset)
    labels = [r.directed_label for r in relationships]
    return set(labels)
