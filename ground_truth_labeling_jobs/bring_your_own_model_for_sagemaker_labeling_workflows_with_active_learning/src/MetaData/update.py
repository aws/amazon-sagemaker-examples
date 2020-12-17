import json

def lambda_handler(event, context):
    """
    This function is used to update the meta_data values based on active learning logic output.
    """
    output_str = event['active_learning_output']
    output_json = json.loads(output_str)
    return output_json['meta_data']

