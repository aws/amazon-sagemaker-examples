import subprocess
import time
from importlib import import_module
import sys
import json
from generate import generate
from utils import get_tokenizer
import os

def test_task(s3_client, 
              sm_runtime_client, 
              model_id: str,
              test_spec: dict,
              endpoint_name: str,
              results_path: str,
              streaming_enabled: bool,
              hf_token:str=None):

    task_name = test_spec.get('task_name', "text-generation")

    if task_name == "text-generation":
        tokenizer = get_tokenizer(s3_client, model_id, hf_token=hf_token)
        return _test_text_generation(sm_runtime_client=sm_runtime_client, 
                       tokenizer=tokenizer, 
                       test_spec=test_spec,
                       endpoint_name=endpoint_name,
                       results_path=results_path,
                       streaming_enabled=streaming_enabled)
    elif task_name == "reranker":
        return _test_reranker(sm_runtime_client=sm_runtime_client,
                              test_spec=test_spec,
                              endpoint_name=endpoint_name,
                              results_path=results_path)
    else:
        raise ValueError(f"Unknown task: {task_name}")


def _test_text_generation(sm_runtime_client, 
                    tokenizer,
                    test_spec: dict, 
                    endpoint_name:str, 
                    results_path: str,
                    streaming_enabled: bool) -> None:
    module_name = test_spec.get('module_name', None)
    assert module_name, "'test.module_name' is required"
    
    module_dir = test_spec.get('module_dir', None)
    assert module_name, "'test.module_dir' is required"
    
    prompt_generator = test_spec.get('prompt_generator', None)
    assert prompt_generator, "'test.prompt_generator' is required"
    
    sys.path.append(module_dir)
    
    requirements_path = os.path.join(module_dir, "requirements.txt")
    if os.path.isfile(requirements_path):
        print(f"Installing test module requirements...")
        subprocess.check_output(f"pip install -r {requirements_path}", shell=True, stderr=subprocess.STDOUT)
    
    print(f"Loading test module: {module_name} from {module_dir}")
    mod=import_module(module_name)
    prompt_generator_class = getattr(mod, prompt_generator)
    
    print(f"Creating prompt generator object for class: {prompt_generator_class}")
    prompt_generator = prompt_generator_class()()

    warmup_iters = int(test_spec.get('warmup_iters', 1))
    max_iters = int(test_spec.get('max_iters', 10))
    params = test_spec.get("params", None)
    input_type = test_spec.get("input_type", "list")
    
    cumu_time = 0.0
    cumu_tokens = 0
    cumu_ttft = 0.0
    
    try:
        with open(results_path, "w") as results:
            count = 0
            
            print("Start testing...")
            while prompt := next(prompt_generator):
                ttft = None
                start_time = time.time()
                
                text, ttft = generate(sm_runtime_client, 
                                      endpoint_name, 
                                      [prompt] if input_type == "list" else prompt, 
                                      params=params, 
                                      stream=streaming_enabled)
                latency = time.time() - start_time
                    
                count += 1
                if count <= warmup_iters:
                    print(f"Warm up iteration: {count} of {warmup_iters}. latency: {latency}, ttft: {ttft}")
                    continue
                
                if ttft:
                    cumu_ttft += ttft
                
                iter_count = count - warmup_iters
                
                cumu_time += latency
                index = text.find(prompt)
                if index != -1:
                    text = text[len(prompt):]
                    
                n_tokens = len(tokenizer.encode(text))
                cumu_tokens += n_tokens
                
                tps = n_tokens/latency
                
                json_obj = {"prompt": prompt, 
                            "text": text, 
                            "n_tokens": n_tokens,
                            "latency": latency, 
                            "tps": tps,
                            "ttft": ttft}
                
                results.write(json.dumps( json_obj )+"\n")   
                avg_latency = cumu_time/iter_count
                avg_tps = cumu_tokens/cumu_time
                avg_tokens = cumu_tokens/iter_count
                avg_ttft = cumu_ttft/iter_count
                
                print(f"Iterations completed: {iter_count} of {max_iters}; avg_tokens: {avg_tokens}, avg_latency: {avg_latency} secs, avg_tps: {avg_tps}, avg_ttft: {avg_ttft}")
                if iter_count >= max_iters:
                    break    
    except StopIteration as e:
        print(f"Error: {e}")

    print(f"Testing completed. Results file: {results_path}")

def _test_reranker(sm_runtime_client, 
                    test_spec: dict, 
                    endpoint_name:str, 
                    results_path: str) -> None:
    module_name = test_spec.get('module_name', None)
    assert module_name, "'test.module_name' is required"
    
    module_dir = test_spec.get('module_dir', None)
    assert module_name, "'test.module_dir' is required"
    
    prompt_generator = test_spec.get('prompt_generator', None)
    assert prompt_generator, "'test.prompt_generator' is required"
    
    sys.path.append(module_dir)
    
    requirements_path = os.path.join(module_dir, "requirements.txt")
    if os.path.isfile(requirements_path):
        print(f"Installing test module requirements...")
        subprocess.check_output(f"pip install -r {requirements_path}", shell=True, stderr=subprocess.STDOUT)
    
    print(f"Loading test module: {module_name} from {module_dir}")
    mod=import_module(module_name)
    prompt_generator_class = getattr(mod, prompt_generator)
    
    print(f"Creating prompt generator object for class: {prompt_generator_class}")
    prompt_generator = prompt_generator_class()()

    warmup_iters = int(test_spec.get('warmup_iters', 1))
    max_iters = int(test_spec.get('max_iters', 10))
    params = test_spec.get("params", None)
    cumu_time = 0.0

    try:
        with open(results_path, "w") as results:
            count = 0
            
            print("Start testing...")
            while prompt := next(prompt_generator):
                start_time = time.time()
                
                data= { "inputs": prompt }
                data["parameters"] = params
                body = json.dumps(data).encode("utf-8")
                response = sm_runtime_client.invoke_endpoint(EndpointName=endpoint_name, 
                                                    ContentType="application/json", 
                                                    Accept="application/json", Body=body)
                latency = time.time() - start_time
                    
                count += 1
                if count <= warmup_iters:
                    print(f"Warm up iteration: {count} of {warmup_iters}. latency: {latency}")
                    continue
                
                iter_count = count - warmup_iters
                cumu_time += latency
                    
                body = response["Body"].read()
                scores = json.loads( body.decode("utf-8"))

                json_obj = {"prompt": prompt, 
                            "scores": scores, 
                            "latency": latency}
                
                results.write(json.dumps( json_obj )+"\n")   
                avg_latency = cumu_time/iter_count
                
                print(f"Iterations completed: {iter_count} of {max_iters}; avg_latency: {avg_latency} secs")
                if iter_count >= max_iters:
                    break
    except StopIteration as e:
        print(f"Error: {e}")    
    print(f"Testing completed. Results file: {results_path}")