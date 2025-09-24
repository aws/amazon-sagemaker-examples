from collections import OrderedDict
from typing import List, Set, Tuple, Union, Optional
import re
from pathlib import Path


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
        assert (
            text[idx + len(token):].find(token) == -1
        ), f"{token} duplicated in text."
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


def parse_file(
    file_path: Union[Path, str],
    sample_idx_offset: Optional[int] = None
) -> List:
    file_path = Path(file_path)
    with open(file_path, "r") as f:
        lines = f.readlines()
    num_lines = len(lines)
    num_samples = int(num_lines / 4)
    assert (
        num_samples * 4 == num_lines
    ), f"Should have 4 lines per block, but {num_lines} cannot be divided by 4."
    examples, ground_truth = [], []
    for sample_idx in range(num_samples):
        block = lines[(sample_idx * 4): ((sample_idx + 1) * 4)]
        entity_one_start, entity_one_end, entity_two_start, entity_two_end, text, label, is_reversed = parse_block(block)
        
        examples.append(
            {
                "sequence": text,
                "entity_one_start": entity_one_start,
                "entity_one_end": entity_one_end,
                "entity_two_start": entity_two_start,
                "entity_two_end": entity_two_end
            
            }
        )
        ground_truth.append(
            label
        )
        
    return examples, ground_truth


def parse_block(block: List[str]):
    text, label, comment, blank = block
    entity_one_start, entity_one_end, entity_two_start, entity_two_end, text = parse_text(text)
    label, is_reversed = parse_label(label)
    
    label_seperator = '-'
    if is_reversed:
        parts = label.split(label_seperator)
        assert len(parts) == 2
        parts.reverse()
        label = label_seperator.join(parts)
    return entity_one_start, entity_one_end, entity_two_start, entity_two_end, text, label, is_reversed


def parse_text(text: str):
    sample_idx_str, text = text.split("\t")
    if '"\n' in text:
        text = text.strip('"\n')
    sample_idx = int(sample_idx_str)
    text = text.strip('"')
    special_tokens = ["<e1>", "</e1>", "<e2>", "</e2>"]
    idxs = find_special_tokens(text, special_tokens)
    text = remove_special_tokens(text, special_tokens)
    return idxs["<e1>"], idxs["</e1>"], idxs["<e2>"], idxs["</e2>"], text


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
        label = label.rstrip('\n')
        if label == 'Other':
            is_reversed = False
        else:
            raise ValueError(f"Cannot find {pattern} in {label}")
    return label, is_reversed
