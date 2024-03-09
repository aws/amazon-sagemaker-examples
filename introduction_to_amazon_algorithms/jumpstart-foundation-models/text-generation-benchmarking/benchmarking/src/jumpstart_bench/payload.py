def create_test_payload(input_words: int, output_tokens: int, details: bool = False):
    inputs = "I believe the meaning of life is to " * (input_words // 8)
    parameters = {"max_new_tokens": output_tokens, "temperature": 99.0}
    if details is True:
        parameters["details"] = True
    return {"inputs": inputs, "parameters": parameters}
