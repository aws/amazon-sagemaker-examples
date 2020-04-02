from ActiveLearning.helper import SimpleActiveLearning
import pytest

def test_compute_margin_high_confidence():
    al = SimpleActiveLearning("test", "animal", ['dog', 'cat'], 1000)
    confidence, chosen = al.compute_margin([0.9,0.1], ['dog', 'cat'])

    assert chosen == 'dog'
    assert confidence == pytest.approx(0.8)

def test_compute_margin_low_confidence():
    al = SimpleActiveLearning("test", "animal", ['dog', 'cat'], 1000)
    confidence, chosen = al.compute_margin([0.6,0.4], ['dog', 'cat'])

    assert chosen == 'dog'
    assert confidence == pytest.approx(0.2)

def test_get_label_index():
    al = SimpleActiveLearning("test", "animal", ['dog', 'cat'], 1000)
    assert al.get_label_index("__label__0") == 0
    assert al.get_label_index("__label__1") == 1

def test_autoannotate():
    al = SimpleActiveLearning("test", "animal", ['dog', 'cat'], 1000)
    sources = [{"id": 0, "source" :"This text is about a dog."},
               {"id": 1, "source": "This text is not about animals."}]
    predictions = [{"id": 0, "prob" : [0.9, 0.1], "label": ["__label__0", "__label__1"]},
                   {"id": 1, "prob": [0.6, 0.4], "label": ["__label__0", "__label__1"]}]

    autoannotations = al.autoannotate(predictions, sources)
    assert len(autoannotations) == 1
    assert autoannotations[0]['id'] == 0
    assert autoannotations[0]['animal-metadata']['class-name'] == 'dog'
    assert autoannotations[0]['animal-metadata']['human-annotated'] == 'no'

def test_select_for_labeling():
    al = SimpleActiveLearning("test", "animal", ['dog', 'cat'], 1000)
    sources = [{"id": 0, "source" :"This text is about a dog."},
               {"id": 1, "source": "This text is not about animals."}]
    predictions = [{"id": 0, "prob" : [0.9, 0.1], "label": ["__label__0", "__label__1"]},
                   {"id": 1, "prob": [0.6, 0.4], "label": ["__label__0", "__label__1"]}]

    autoannotations = al.autoannotate(predictions, sources)
    selected = al.select_for_labeling(predictions,autoannotations)
    assert len(selected) == 1
    assert selected[0] == 1
