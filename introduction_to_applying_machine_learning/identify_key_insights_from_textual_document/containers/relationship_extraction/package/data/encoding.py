import json
from typing import Union

from tokenizers import Encoding


def get_state(encoding: Encoding) -> dict:
    """
    Given an encoding output from a tokenizer, this function will return
    its internal state as a dictionary. You can modify this state and then
    use `from_state` to generate an new Encoding object.

    Args:
        encoding (Encoding): encoding output from tokenizer

    Returns:
        dict: state of encoding
    """
    # json.loads(encoding.__getstate__()) is x3 slower
    return {
        "ids": encoding.ids,
        "type_ids": encoding.type_ids,
        "tokens": encoding.tokens,
        "words": encoding.words,
        "offsets": encoding.offsets,
        "special_tokens_mask": encoding.special_tokens_mask,
        "attention_mask": encoding.attention_mask,
        "overflowing": encoding.overflowing,
    }


def from_state(state: dict) -> Encoding:
    """
    Given a dictionary containing the state of an encoding, this function
    will return an associated Encoding object. Use `get_state` to obtain
    the state from an Encoding object.

    Args:
        state (dict): state of encoding

    Returns:
        Encoding: encoding
    """
    encoding = Encoding()
    encoding.__setstate__(json.dumps(state).encode())
    return encoding


def char_to_next_token(encoding: Encoding, char_idx: int) -> Union[int, None]:
    """
    Given an encoding output from a tokenizer, this function will return
    the token index that corresponds to a given character index. When a
    character index doesn't correspond to a token (e.g. a white space), the
    next avaliable token index is returned. If there is no next avaliable
    token index, this function returns None. Used to find the token index
    where entity start tokens should be inserted.

    Args:
        encoding (Encoding): encoding output from tokenizer
        char_idx (int): character index of input sequence given to tokenizer

    Returns:
        Union[int, None]: corresponding token index
    """
    length = max([end for start, end in encoding.offsets])
    assert char_idx >= 0
    token_idx = encoding.char_to_token(char_idx)
    while token_idx is None:
        char_idx += 1
        if char_idx > length:
            return None
        token_idx = encoding.char_to_token(char_idx)
    return token_idx


def char_to_previous_token(encoding: Encoding, char_idx: int) -> Union[int, None]:
    """
    Given an encoding output from a tokenizer, this function will return
    the token index that corresponds to a given character index. When a
    character index doesn't correspond to a token (e.g. a white space), the
    previous token index (that is not None) is returned. If there is no
    previous token index that is not None, this function returns None. Used
    to find the token index where entity end tokens should be inserted.

    Args:
        encoding (Encoding): encoding output from tokenizer
        char_idx (int): character index of input sequence given to tokenizer

    Returns:
        Union[int, None]: corresponding token index
    """
    assert char_idx >= 0
    token_idx = encoding.char_to_token(char_idx)
    while token_idx is None:
        char_idx -= 1
        if char_idx < 0:
            return None
        token_idx = encoding.char_to_token(char_idx)
    return token_idx
