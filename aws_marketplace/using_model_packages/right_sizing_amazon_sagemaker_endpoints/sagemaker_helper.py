import boto3
sm_client = boto3.client('sagemaker')

import concurrent.futures

endpoints = []

def deploy_single_endpoint(item, cpu_model=None, gpu_model=None):
    """
    Deploys a single SageMaker endpoint
    
    This function deploys endpoint based on a single item and models
    with the name 'endpoint-<type>-<count>' and adds endpoint name to 
    the global list.
    
    Inputs:
    endpoints_dict: dict of instance types and counts to deploy
    cpu_model: CPU model ARN, if model supports CPU
    gpu_model: GPU model ARN, if model supports GPU
    
    Output:
    None   
    """
    instance_type = item.get("instance_type")
    instance_count = item.get("instance_count")
    existing_endpoints = get_existing_endpoints()

    endpoint_name = f"endpoint-{instance_type.replace('.', '-')}-x{instance_count}"

    # Check if endpoint already exists
    if endpoint_name in existing_endpoints:
        print(f'Endpoint {endpoint_name} already exists.')
        return endpoint_name

    else:
        if instance_type.split('.')[1].startswith(('m', 'c', 'r')):
            if cpu_model:
                print(f"\nDeploying to {endpoint_name}...")
                predictor = cpu_model.deploy(instance_count, instance_type, endpoint_name=endpoint_name, wait=True)
            else:
                print(f"No CPU model specified for a CPU instance of type {instance_type}")

            # gpu instance types
        elif instance_type.split('.')[1].startswith(('p', 'g', 'e', 'i')):
            if gpu_model:
                print(f"\nDeploying to {endpoint_name}...")
                predictor = gpu_model.deploy(instance_count, instance_type, endpoint_name=endpoint_name, wait=True)
            else:
                print(f"No GPU model specified for a GPU instance of type {instance_type}")
        else:
            print(f"Unsupported instance type {instance_type}")
            
        return endpoint_name

            
def deploy_endpoints(endpoints_dict, cpu_model=None, gpu_model=None):
    """
    Deploys multiple endpoints concurrently by calling deploy_single_endpoint()
    """
    endpoints = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for item in endpoints_dict:
            futures.append(executor.submit(deploy_single_endpoint, item=item, cpu_model=cpu_model, gpu_model=gpu_model))

        for future in concurrent.futures.as_completed(futures):
            endpoints.append(future.result())

    return endpoints


def clean_up_endpoints(endpoints_list):
    """
    Deletes a given list of endpoints
    
    This function checks if a given endpoint exists, and if yes, 
    deletes the endpoint.
    
    Input:
    endpoints_list: list of endpoints to delete
    
    Output:
    None
    """
    endpoints = get_existing_endpoints()
    endpoints_list = list(set(endpoints_list))  # avoiding duplicates
    for endpoint in endpoints_list:
        if endpoint in endpoints:
            try:
                print(f'Deleting {endpoint}..')
                sm_client.delete_endpoint(
                    EndpointName=endpoint
                )
            except Exception as e:
                print(f"Error deleting endpoint {endpoint}: {e}")
        

def get_existing_endpoints():
    """
    Returns list of existing SageMaker endpoint names
    """
    response = sm_client.list_endpoints()
    endpoints = response['Endpoints']
    if endpoints:
        endpoint_names = [item['EndpointName'] for item in endpoints]
    else:
        endpoint_names = []
        
    return endpoint_names