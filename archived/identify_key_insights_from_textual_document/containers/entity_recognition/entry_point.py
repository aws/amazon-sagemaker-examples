import json
import spacy
import subprocess


def extract_entities(spacy_document):
    entities = []
    for spacy_entity in spacy_document.ents:
        entity = {
            'text': spacy_entity.text,
            'start_char': spacy_entity.start_char,
            'end_char': spacy_entity.end_char,
            'label': spacy_entity.label_
        }
        entities.append(entity)
    return entities


def extract_noun_chunks(spacy_document):
    noun_chunks = []
    for spacy_noun_chunk in spacy_document.noun_chunks:
        noun_chunk = {
            'text': spacy_noun_chunk.text,
            'start_char': spacy_noun_chunk.start_char,
            'end_char': spacy_noun_chunk.end_char
        }
        noun_chunks.append(noun_chunk)
    return noun_chunks


def model_fn(model_dir):
    spacy_model = "en_core_web_md"
    subprocess.run(f"python -m spacy download {spacy_model}", shell=True)
    nlp = spacy.load(spacy_model)
    model_assets = {
        "nlp": nlp
    }
    return model_assets


def input_fn(request_body_str, request_content_type):
    assert (
        request_content_type == "application/json"
    ), "content_type must be 'application/json'"
    request_body = json.loads(request_body_str)
    return request_body


def get_parameter(request_body, parameter_name, default):
    parameter = default
    if 'parameters' in request_body:
        if parameter_name in request_body['parameters']:
            parameter = request_body['parameters'][parameter_name]
    return parameter


def predict_fn(request_body, model_assets):
    nlp = model_assets['nlp']
    text = request_body["text"]
    spacy_document = nlp(text)
    entities = extract_entities(spacy_document)
    noun_chunks = extract_noun_chunks(spacy_document)
    return {
        "entities": entities,
        "noun_chunks": noun_chunks
    }


def output_fn(prediction, response_content_type):
    assert (
        response_content_type == "application/json"
    ), "accept must be 'application/json'"
    response_body_str = json.dumps(prediction)
    return response_body_str
