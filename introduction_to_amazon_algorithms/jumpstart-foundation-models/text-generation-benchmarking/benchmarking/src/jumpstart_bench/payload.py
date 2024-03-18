from typing import Any, Dict

from transformers import PreTrainedTokenizer


def create_test_payload(
    input_words: int, output_tokens: int, details: bool = False, stream: bool = False, set_high_temperature: bool = True
) -> Dict[str, Any]:
    """Returns a simple test payload given number of input words and output tokens."""
    inputs = "I believe the meaning of life is to " * (input_words // 8)
    kwargs = create_test_payload_args(output_tokens, details, stream, set_high_temperature)
    return {"inputs": inputs, **kwargs}


def create_test_payload_args(
    output_tokens: int, details: bool = False, stream: bool = False, set_high_temperature: bool = True
) -> Dict[str, Any]:
    """Returns the non-input arguments for a simple test payload given number output tokens."""
    parameters = {"max_new_tokens": output_tokens}
    if set_high_temperature is True:
        parameters["temperature"] = 99.0
    if details is True:
        parameters["details"] = True
    return {"parameters": parameters, "stream": stream}


def add_items_to_dict(example: Dict[str, Any], items: Dict[str, Any]) -> Dict[str, Any]:
    example.update(items)
    return example


def apply_chat_template(example: Dict[str, Any], tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
    example["inputs"] = tokenizer.apply_chat_template(
        example.pop("messages"), tokenize=False, add_generation_prompt=True
    )
    return example
