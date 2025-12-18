import json
import io
import time

class LineIterator:

    def __init__(self, stream):
        self.byte_iterator = iter(stream)
        self.buffer = io.BytesIO()
        self.read_pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            self.buffer.seek(self.read_pos)
            line = self.buffer.readline()
            if line and line[-1] == ord('\n'):
                self.read_pos += len(line)
                return line[:-1]
            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    continue
                raise
            if 'PayloadPart' not in chunk:
                print('Unknown event type:' + chunk)
                continue
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk['PayloadPart']['Bytes'])

def __invoke_endpoint(client, endpoint_name:str, data: dict): 
    body = json.dumps(data).encode("utf-8")
    response = client.invoke_endpoint(EndpointName=endpoint_name, 
                                    ContentType="application/json", 
                                    Accept="application/json", Body=body)
    body = response["Body"].read()
    json_str = body.decode("utf-8")
    json_obj = json.loads(json_str)
    return json_obj

def __invoke_streaming_endpoint(client, endpoint_name:str, data: dict):    
    body = json.dumps(data).encode("utf-8")
    response = client.invoke_endpoint_with_response_stream(EndpointName=endpoint_name, 
                                    ContentType="application/json", 
                                    Accept="application/json", Body=body)
    event_stream = response['Body']
    return event_stream


def __generate(client, endpoint_name:str, data: dict):
    json_obj = __invoke_endpoint(client, endpoint_name=endpoint_name, data=data)
    json_obj_type = type(json_obj)
    while json_obj_type is list:
        json_obj = json_obj[0]
        json_obj_type = type(json_obj)
    
    generated_text = ''
    
    if json_obj_type is dict:
        # Handle OpenAI chat completions format
        if 'choices' in json_obj and len(json_obj['choices']) > 0:
            choice = json_obj['choices'][0]
            if 'message' in choice and 'content' in choice['message']:
                generated_text = choice['message']['content']
            elif 'text' in choice:
                generated_text = choice['text']
            else:
                raise RuntimeError(f"Unexpected output: {json_obj}")
        # Handle other formats
        elif 'outputs' in json_obj:
            output = json_obj['outputs']
        elif 'generated_text' in json_obj:
            output = json_obj['generated_text']
        elif 'text_output' in json_obj:
            output = json_obj['text_output']
        else:
            raise RuntimeError(f"Unexpected output: {json_obj}")
        
        # Process output if not already set by OpenAI format
        if not generated_text:
            if type(output) is list:
                output = output[0]
            generated_text = output
    else:
        generated_text = json_obj

    return generated_text, None

def __generate_streaming(client, endpoint_name:str, data: dict):

    event_stream = __invoke_streaming_endpoint(client, endpoint_name=endpoint_name, data=data)
    start_time = time.time()
    ttft = None
    generated_text =  ''

    n_tokens = 0
    for json_line in LineIterator(event_stream):

        json_obj = json.loads(json_line)
        if "token" in json_obj and "text" in json_obj["token"]:
            n_tokens += 1
            if ttft is None:
                ttft = time.time() - start_time
            generated_text += json_obj["token"]["text"]
                
    return generated_text, ttft

def generate(client, endpoint_name:str, data: dict, stream:bool):
    if not stream:
        return __generate(client=client, endpoint_name=endpoint_name, data=data)
    else:
        return __generate_streaming(client=client, endpoint_name=endpoint_name, data=data)
    
    