import json
from typing import Optional
from typing import Union


class Source(object):
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

    @classmethod
    def from_dict(cls, args_dict: dict) -> Union["Source", None]:
        if args_dict:
            return cls(**args_dict)
        else:
            return None


class Statement(object):
    def __init__(self, text: str, start_char: int, end_char: int):
        self.start_char = start_char
        self.end_char = end_char
        self.text = text
        self._validate()

    def _validate(self):
        assert self.start_char >= 0
        assert self.end_char > self.start_char
        assert (self.end_char - self.start_char) == len(self.text)

    def __str__(self):
        return self.text

    def __len__(self):
        return len(self.text)

    @classmethod
    def from_dict(cls, args_dict: dict) -> Union["Statement", None]:
        if args_dict:
            return cls(**args_dict)
        else:
            return None


class Entity:
    def __init__(
        self,
        text: str,
        start_char: int,
        end_char: int,
        label: Optional[str] = None,
        kb_id: Optional[str] = None,
    ) -> None:

        self.text = text
        self.start_char = start_char
        self.end_char = end_char
        self.label = label
        self.kb_id = kb_id
        self._validate()

    def _validate(self):
        assert self.start_char >= 0
        assert self.end_char > self.start_char
        assert (self.end_char - self.start_char) == len(self.text)

    def __str__(self):
        return self.text

    def __len__(self):
        return len(self.text)

    @classmethod
    def from_dict(cls, args_dict: dict) -> "Entity":
        return cls(**args_dict)

    # todo: needs to be blanked out at random rate
    # and probably need to know if this has happened (or could just check blank token id)


class Relationship:
    def __init__(
        self,
        entity_one: Entity,
        entity_two: Entity,
        statement: Optional[Statement] = None,
        source: Optional[Source] = None,
        label: Optional[str] = None,
        label_seperator: Optional[str] = None,
        is_reversed: Optional[bool] = False,
    ) -> None:
        self.entity_one = entity_one
        self.entity_two = entity_two
        self.statement = statement
        self.source = source
        self.label = label
        self.label_seperator = label_seperator
        self.is_reversed = is_reversed
        self._validate()

    def _validate(self):
        if self.statement:
            s_text = self.statement.text
            e1_text = self.entity_one.text
            e1_start = self.entity_one.start_char
            e1_end = self.entity_one.end_char
            e1_idxs = set(range(e1_start, e1_end))
            e2_text = self.entity_two.text
            e2_start = self.entity_two.start_char
            e2_end = self.entity_two.end_char
            e2_idxs = set(range(e2_start, e2_end))
            assert e1_end <= len(s_text)
            assert e2_end <= len(s_text)
            assert s_text[e1_start:e1_end] == e1_text
            assert s_text[e2_start:e2_end] == e2_text
            assert len(e1_idxs.intersection(e2_idxs)) == 0

    def to_json_str(self):
        return json.dumps(self, default=lambda x: x.__dict__)

    @classmethod
    def from_dict(cls, args_dict: dict) -> Union["Relationship", None]:
        return cls(
            entity_one=Entity.from_dict(args_dict["entity_one"]),
            entity_two=Entity.from_dict(args_dict["entity_two"]),
            statement=Statement.from_dict(args_dict["statement"]),
            source=Source.from_dict(args_dict["source"]),
            label=args_dict["label"],
            is_reversed=args_dict["is_reversed"],
        )

    @property
    def directed_label(self):
        assert isinstance(self.label_seperator, str)
        if self.is_reversed:
            parts = self.label.split(self.label_seperator)
            assert len(parts) == 2
            parts.reverse()
            return self.label_seperator.join(parts)
        else:
            return self.label
