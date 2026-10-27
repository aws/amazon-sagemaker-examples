import random
from collections import namedtuple
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

from package.data.encoding import char_to_next_token
from package.data.encoding import char_to_previous_token
from package.data.encoding import get_state
from tokenizers import Encoding
from tokenizers import InputSequence
from tokenizers import Tokenizer
from transformers import AutoTokenizer


class EntityToken:
    def __init__(self, token: str, token_id: int, token_idx: int) -> None:
        self.token = token
        self.token_id = token_id
        self.token_idx = token_idx


class RelationshipTokenizer:

    ENTITY_TOKENS = {
        "entity_one_start_token": "[E1]",
        "entity_one_end_token": "[/E1]",
        "entity_two_start_token": "[E2]",
        "entity_two_end_token": "[/E2]",
        "blank_token": "[BLANK]",
    }

    def __init__(
        self, tokenizer: Tokenizer, contains_entity_tokens: bool = True, output_length: Optional[int] = None, **kwargs
    ):
        self.tokenizer = tokenizer
        if contains_entity_tokens:
            vocab = self.tokenizer.get_vocab()
            for entity_token in self.ENTITY_TOKENS.values():
                assert entity_token in vocab, f"{entity_token} not found in tokenizer vocab."
        else:
            num_added = self.tokenizer.add_special_tokens(list(self.ENTITY_TOKENS.values()))
            num_pre_exist = num_added - len(self.ENTITY_TOKENS)
            assert num_pre_exist == 0, f"{num_pre_exist} special token(s) already found in tokenizer vocab."
        if output_length:
            self.set_truncation(output_length)
            self.set_padding(output_length)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "RelationshipTokenizer":
        """
        Used to initialize from a transformers pre-trained tokenizer.

        Args:
            pretrained_model_name_or_path (str):
                name or path to transformers model
                (e.g. 'bert-base-uncased')
            **kwargs:
                keyword arguments are passed to AutoTokenizer.from_pretrained(...)
                and RelationshipTokenizer constructor
                (e.g. can pass 'contains_entity_tokens').

        Returns:
            RelationshipTokenizer:
                relationship tokenizer which wraps tokenizers.Tokenizer
        """
        if "use_fast" not in kwargs:
            kwargs["use_fast"] = True
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        # using base_tokenizer to get tokenizers.Tokenizer object.
        return cls(tokenizer=tokenizer._tokenizer, **kwargs)

    def save(self, file_path: Union[str, Path], pretty: bool = False) -> str:
        """
        Save the tokenizer as JSON formatted file. Optionally prettified.

        Args:
            file_path (str): file path to save tokenizer
            pretty (bool, optional): format JSON with indentation

        Returns:
            str: file path of saved tokenizer
        """
        return self.tokenizer.save(str(file_path), pretty)

    @classmethod
    def from_file(cls, file_path: Union[str, Path], **kwargs) -> "RelationshipTokenizer":
        """
        Creates a tokenizer from a valid JSON formatted file.

        Args:
            file_path (str): file path of saved tokenizer

        Returns:
            RelationshipTokenizer: tokenizer loaded from file
        """
        tokenizer = Tokenizer.from_file(str(file_path))
        return cls(tokenizer=tokenizer, **kwargs)

    def encode(
        self,
        sequence: InputSequence,
        entity_one_start: int,
        entity_one_end: int,
        entity_two_start: int,
        entity_two_end: int,
        blank_entity_proba: float = 0.0,
    ) -> Encoding:
        """
        Given a textual statement containing two entities (i.e. a
        relationship statement), this method tokenizes the statement and
        optionally adds all of the necessary special tokens. Some of the
        special tokens are taken from the underlying tokenizer (e.g. [CLS]
        and [SEP]), while others are added afterwards to locate the
        entities (e.g. [E1], [/E1], [E2] and [/E2] as defined in
        ENTITY_TOKENS).

        Args:
            sequence (InputSequence): textual statement (which is often just a string).
            entity_one_start (int): character index for the start of entity one
            entity_one_end (int): character index for the end of entity one
            entity_two_start (int): character index for the start of entity two
            entity_two_end (int): character index for the end of entity two
            blank_entity_proba (int): probability of blanking out each entity

        Returns:
            Encoding: encoding containing tokens, token ids, special token mask, etc.
        """
        encoding = self.tokenizer.encode(sequence=sequence)
        encoding = self.add_entity_tokens(
            encoding, entity_one_start, entity_one_end, entity_two_start, entity_two_end, blank_entity_proba
        )
        return encoding

    def add_entity_tokens(
        self,
        encoding: Encoding,
        entity_one_start: int,
        entity_one_end: int,
        entity_two_start: int,
        entity_two_end: int,
        blank_entity_proba: float,
    ) -> Encoding:
        """
        Adds special entity tokens to locate the entities (e.g. [E1],
        [/E1], [E2] and [/E2] as defined in ENTITY_TOKENS).

        Args:
            encoding (Encoding): encoding with underlying special tokens (e.g. [CLS] and [SEP])
            entity_one_start (int): character index for the start of entity one
            entity_one_end (int): character index for the end of entity one
            entity_two_start (int): character index for the start of entity two
            entity_two_end (int): character index for the end of entity two
            blank_entity_proba (int): probability of blanking out each entity

        Returns:
            Encoding: encoding with additional entity tokens
        """
        max_idx = max([e for span in encoding.offsets for e in span])
        assert (0 <= entity_one_start) and (entity_one_start <= max_idx)
        assert (0 <= entity_one_end) and (entity_one_end <= max_idx)
        assert (0 <= entity_two_start) and (entity_two_start <= max_idx)
        assert (0 <= entity_two_end) and (entity_two_end <= max_idx)
        entity_tokens = []
        entity_tokens.append(
            EntityToken(
                token=self.entity_one_start_token,
                token_id=self.entity_one_start_token_id,
                token_idx=char_to_next_token(encoding, entity_one_start),
            )
        )
        entity_tokens.append(
            EntityToken(
                token=self.entity_one_end_token,
                token_id=self.entity_one_end_token_id,
                token_idx=char_to_previous_token(encoding, entity_one_end - 1) + 1,
            )
        )
        entity_tokens.append(
            EntityToken(
                token=self.entity_two_start_token,
                token_id=self.entity_two_start_token_id,
                token_idx=char_to_next_token(encoding, entity_two_start),
            )
        )
        entity_tokens.append(
            EntityToken(
                token=self.entity_two_end_token,
                token_id=self.entity_two_end_token_id,
                token_idx=char_to_previous_token(encoding, entity_two_end - 1) + 1,
            )
        )
        entity_tokens = sorted(entity_tokens, key=lambda t: t.token_idx)
        self.assert_entity_token_ordering(entity_tokens)
        state = get_state(encoding)
        for offset, entity_token in enumerate(entity_tokens):
            # token_idx is relative to the original encoding. using offset
            # to correct for other entity tokens that have added earlier in loop.
            entity_token.token_idx += offset
            self.add_entity_token_to_state(state, entity_token)
        if blank_entity_proba > 0:
            self.blank_entities(state, entity_tokens, blank_entity_proba)
        return state

    def assert_entity_token_ordering(self, entity_tokens: List[EntityToken]) -> None:
        """
        Asserts that entities don't overlap and that entity start comes
        before entity end (for both entity one and two).
        Note: it's permitted that entity two comes before entity one.

        Args:
            entity_tokens (List[EntityToken]):
                pre-sorted list of entity tokens.
                pre-sorted by token index in ascending order.
        Raises:
            AssertionError: when entities overlap or start/stop are out of order.
        """
        tokens = [e.token for e in entity_tokens]
        first_token = tokens[0]
        if first_token == self.entity_one_start_token:
            assert tokens == [
                self.entity_one_start_token,
                self.entity_one_end_token,
                self.entity_two_start_token,
                self.entity_two_end_token,
            ]
        elif first_token == self.entity_two_start_token:
            assert tokens == [
                self.entity_two_start_token,
                self.entity_two_end_token,
                self.entity_one_start_token,
                self.entity_one_end_token,
            ]
        else:
            raise AssertionError(f"{first_token} is not {self.entity_one_start_token} or {self.entity_two_start_token}")

    @classmethod
    def add_entity_token_to_state(cls, state: dict, entity_token: EntityToken) -> None:
        """
        Adds a single entity token to the state of an encoding.

        Args:
            state (dict):
                other entity tokens have already been added to state
            entity_token (EntityToken):
                token, token_id and token_idx are taken from entity_token
        """
        idx = entity_token.token_idx
        state["ids"].insert(idx, entity_token.token_id)
        state["type_ids"].insert(idx, 0)
        state["tokens"].insert(idx, entity_token.token)
        state["words"].insert(idx, None)
        state["offsets"].insert(idx, [0, 0])
        state["special_tokens_mask"].insert(idx, 1)
        state["attention_mask"].insert(idx, 1)

    def blank_entities(self, state: dict, entity_tokens: List[EntityToken], blank_entity_proba: float) -> None:
        """
        Given a valid and pre-ordered list of entity tokens, this method
        will randomly blank out each entity with a probability of
        `blank_entity_proba`. When an entity is blanked out, all the tokens
        for that entity are replaced with a corresponding number of
        specific blank tokens (see ENTITY_TOKENS).

        Args:
            state (dict):
                state of encoding from tokenizer, with entity tokens added.
            entity_tokens (List[EntityToken]):
                used to determine the start and end of the two entities.
            blank_entity_proba (float):
                probability of blanking out each token, so the probability of
                blanking both tokens is blank_entity_proba ** 2.
        """
        self.blank_entity(
            state=state,
            start_idx=entity_tokens[0].token_idx + 1,
            end_idx=entity_tokens[1].token_idx,
            blank_entity_proba=blank_entity_proba,
        )
        self.blank_entity(
            state=state,
            start_idx=entity_tokens[2].token_idx + 1,
            end_idx=entity_tokens[3].token_idx,
            blank_entity_proba=blank_entity_proba,
        )

    def blank_entity(self, state: dict, start_idx: int, end_idx: int, blank_entity_proba: float):
        """
        Given a start and end token index of an entity, this method will
        randomly blank out the entity with a probability of
        `blank_entity_proba`. When the entity is blanked out, all the
        tokens for that entity are replaced with a corresponding number of
        specific blank tokens (see ENTITY_TOKENS).

        Args:
            state (dict):
                state of encoding from tokenizer with entity tokens added.
            start_idx (int):
                token index of entity start (but after entity start token)
            end_idx (int):
                token index of entity end (but before entity end token)
            blank_entity_proba (float):
                probability of blanking out the token
        """
        if random.uniform(0, 1) < blank_entity_proba:
            for i in range(start_idx, end_idx):
                state["ids"][i] = self.blank_token_id
                state["type_ids"][i] = 0
                state["tokens"][i] = self.blank_token
                state["words"][i] = None
                state["offsets"][i] = [0, 0]
                state["special_tokens_mask"][i] = 1
                state["attention_mask"] = 1

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(ids=ids, skip_special_tokens=skip_special_tokens)

    @property
    def num_entity_tokens_to_add(self) -> int:
        return 4

    @property
    def num_special_tokens_to_add(self) -> int:
        return self.tokenizer.num_special_tokens_to_add(is_pair=False) + self.num_entity_tokens_to_add

    def set_truncation(self, length: int) -> None:
        self.tokenizer.enable_truncation(max_length=length - self.num_entity_tokens_to_add)

    @property
    def truncation_length(self) -> Union[int, None]:
        params = self.tokenizer.truncation
        if params:
            return params["max_length"] + self.num_entity_tokens_to_add
        else:
            return None

    def set_padding(self, length: int) -> None:
        self.tokenizer.enable_padding(length=length - self.num_entity_tokens_to_add)

    @property
    def padding_length(self) -> Union[int, None]:
        params = self.tokenizer.padding
        if params:
            return params["length"] + self.num_entity_tokens_to_add
        else:
            return None

    @property
    def entity_one_start_token(self) -> str:
        return self.ENTITY_TOKENS["entity_one_start_token"]

    @property
    def entity_one_start_token_id(self) -> int:
        return self.tokenizer.token_to_id(self.entity_one_start_token)

    @property
    def entity_one_end_token(self) -> str:
        return self.ENTITY_TOKENS["entity_one_end_token"]

    @property
    def entity_one_end_token_id(self) -> int:
        return self.tokenizer.token_to_id(self.entity_one_end_token)

    @property
    def entity_two_start_token(self) -> str:
        return self.ENTITY_TOKENS["entity_two_start_token"]

    @property
    def entity_two_start_token_id(self) -> int:
        return self.tokenizer.token_to_id(self.entity_two_start_token)

    @property
    def entity_two_end_token(self) -> str:
        return self.ENTITY_TOKENS["entity_two_end_token"]

    @property
    def entity_two_end_token_id(self) -> int:
        return self.tokenizer.token_to_id(self.entity_two_end_token)

    @property
    def blank_token(self) -> str:
        return self.ENTITY_TOKENS["blank_token"]

    @property
    def blank_token_id(self) -> int:
        return self.tokenizer.token_to_id(self.blank_token)

    def __len__(self) -> int:
        return self.tokenizer.get_vocab_size()
