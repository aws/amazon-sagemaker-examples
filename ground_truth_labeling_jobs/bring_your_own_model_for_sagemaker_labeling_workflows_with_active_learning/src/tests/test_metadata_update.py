from MetaData.update import lambda_handler

def test_metadata_update():
    event = {
        'active_learning_output':'{"LabelAttributeName":"category","meta_data":{"counts":{"input_total":10000}}}'
    }

    output = lambda_handler(event, {})

    assert output["counts"]["input_total"] == 10000