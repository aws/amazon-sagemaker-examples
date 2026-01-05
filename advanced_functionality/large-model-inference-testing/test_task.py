import time
import json
from generate import generate
from inference import inference
from utils import (
    get_tokenizer, 
    fill_template, 
    init_prompt_generator,
    set_prompt_env
)
import os
import boto3
import multiprocessing as mp

def test_task(model_id: str,
              test_spec: dict,
              endpoint_name: str,
              results_path: str,
              streaming_enabled: bool=False,
              hf_token:str=None) -> None:

    task_name = test_spec.get('task_name', "text-generation")
    set_prompt_env(test_spec=test_spec)
        
    n_concurrent = test_spec.get('n_concurrent', 1)
    task_args = {}
    if task_name == "text-generation":
        task_fn = _test_text_generation 
        task_args = {
            "model_id": model_id,
            "hf_token": hf_token,
            "test_spec": test_spec,
            "endpoint_name": endpoint_name,
            "results_path": results_path,
            "streaming_enabled": streaming_enabled
        }
    else:
        task_fn = _test_inference
        task_args = {
            "test_spec": test_spec,
            "endpoint_name": endpoint_name,
            "results_path": results_path,
            "task_name": task_name
        }

    os.makedirs(results_path, exist_ok=True)
    print(f"Running {task_name} test with {n_concurrent} concurrent processes...")
    pool = mp.Pool(n_concurrent)
    pool.map(__call_fn, [ (task_fn, task_args) ] * n_concurrent)
    pool.close()
    pool.join()
    

def __call_fn(t: tuple):
    fn = t[0]
    kwargs = t[1]
    fn(**kwargs)

def _test_text_generation(model_id:str, 
                    hf_token: str,
                    test_spec: dict, 
                    endpoint_name:str, 
                    results_path: str,
                    streaming_enabled: bool) -> None:
    
    pid = os.getpid()

    sm_runtime_client = boto3.client("runtime.sagemaker")
    s3_client = boto3.client('s3')
    tokenizer = get_tokenizer(s3_client, model_id, hf_token=hf_token)
    
    prompt_generator, prompt_template, prompt_template_keys = init_prompt_generator()

    warmup_iters = int(test_spec.get('warmup_iters', 1))
    max_iters = int(test_spec.get('max_iters', 10))

    cumu_time = 0.0
    cumu_tokens = 0
    cumu_ttft = 0.0
    
    try:
        ts = round(time.time() * 1000)
        results_path = os.path.join(results_path, f"results-{pid}-{ts}.json")
        with open(results_path, "w") as results:
            count = 0
            
            print(f"{pid}: Start testing...")
            while prompt := next(prompt_generator):
                ttft = None
                
                start_time = time.time()
                data = fill_template(template=prompt_template, template_keys=prompt_template_keys, inputs=prompt)
                text, ttft = generate(sm_runtime_client, endpoint_name, data=data, stream=streaming_enabled)
                prompt = prompt[0] if len(prompt) == 1 else prompt
                index = text.find(prompt) if isinstance(prompt, str) else -1
                if index != -1:
                    text = text[len(prompt):]
                latency = time.time() - start_time
                
                if ttft is None:
                    ttft = latency
                    
                count += 1
                if count <= warmup_iters:
                    print(f"{pid}: Warm up iteration: {count} of {warmup_iters}. latency: {latency}, ttft: {ttft}")
                    continue
                
                if ttft:
                    cumu_ttft += ttft
                
                iter_count = count - warmup_iters
                
                cumu_time += latency   
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
                
                print(f"{pid}: Iterations completed: {iter_count} of {max_iters}; avg_tokens: {avg_tokens}, avg_latency: {avg_latency} secs, avg_tps: {avg_tps}, avg_ttft: {avg_ttft}")
                if iter_count >= max_iters:
                    break    
    except StopIteration as e:
        print(f"{pid}: Error: {e}")

    print(f"{pid}: Testing completed. Results file: {results_path}")

def _test_inference(test_spec: dict, 
                    endpoint_name:str, 
                    results_path: str,
                    task_name: str = "embeddings") -> None:
    
    pid = os.getpid()
    sm_runtime_client = boto3.client("runtime.sagemaker")
    prompt_generator, prompt_template, prompt_template_keys = init_prompt_generator()

    warmup_iters = int(test_spec.get('warmup_iters', 1))
    max_iters = int(test_spec.get('max_iters', 10))
    cumu_time = 0.0

    try:
        ts = round(time.time() * 1000)
        results_path = os.path.join(results_path, f"results-{pid}-{ts}.json")
        with open(results_path, "w") as results:
            count = 0
            
            print(f"{pid}: Start testing...")
            while prompt := next(prompt_generator):

                start_time = time.time()
                data = fill_template(template=prompt_template, template_keys=prompt_template_keys, inputs=prompt)
                
                result = inference(sm_runtime_client, endpoint_name, data=data, task=task_name)
                latency = time.time() - start_time
                    
                count += 1
                if count <= warmup_iters:
                    print(f"{pid}: Warm up iteration: {count} of {warmup_iters}. latency: {latency}")
                    continue
                
                iter_count = count - warmup_iters
                cumu_time += latency
                    
                prompt = prompt[0] if len(prompt) == 1 else prompt
                result = {"prompt": prompt, "output": result, "latency": latency}
                
                results.write(json.dumps( result )+"\n")   
                avg_latency = cumu_time/iter_count
                
                print(f"{pid}: Iterations completed: {iter_count} of {max_iters}; avg_latency: {avg_latency} secs")
                if iter_count >= max_iters:
                    break
    except StopIteration as e:
        print(f"{pid}: Error: {e}")    
    print(f"{pid}: Testing completed. Results file: {results_path}")

