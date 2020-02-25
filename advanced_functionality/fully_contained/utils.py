import boto3
import os
import sagemaker
    
def deploy(script_name, image_version, model_base_name):
    
    base = '{}-{}'.format(model_base_name, image_version)
    
    details = get_script_details(script_name)
    
    add_to_file(details, 'container/base_predictor.py', 'container/scripts/predictor.py')

    image_name = build_push('scripts', base)
    
    create_deploy_endpoint(base, image_name)
    
    return

def build_push(local_folder, image_version):
    local_folder = 'scripts'

    sess = boto3.session.Session()
       
    # get account
    account_id = boto3.client('sts').get_caller_identity()['Account']
    
    # get region
    region = sess.region_name
    
    # get ecr creds 
    os.system('$(aws ecr get-login --region {} --no-include-email)'.format(region))

    # try to create the repository, otherwise pass
    os.system('''aws ecr create-repository --repository-name '{}' || echo Already created'''.format(local_folder))
    
    full_name = "{}.dkr.ecr.{}.amazonaws.com/{}".format(account_id, region, local_folder)

    # handle permissions
    os.system('cd container && chmod +x {}/train && chmod +x {}/serve'.format(local_folder, local_folder))

    # do the local build
    os.system('cd container && docker build -t {} .'.format(local_folder))

    # print the push 
    cmd = 'cd container && docker tag {} {}:{}'.format(local_folder, full_name, image_version)
    print (cmd)
    os.system(cmd)
    
    # push the image
    cmd2 = 'cd container && docker push {}:{}'.format(full_name, image_version)
    print (cmd2)
    os.system(cmd2)
    
    return '{}:{}'.format(full_name, image_version)

def get_script_details(script_name):
    rt = {}

    with open(script_name) as f:
        for row in f.readlines():
            if len(row) >= 2:
                data = row.strip().split('=')
                if 'model' in data[0]:
                    rt['cls.model'] = data[1][1:]
                    
                elif 'transformer' in data[0]:
                    rt['cls.transformer'] = data[1][1:]
                    
    print ('Got the details from your script,', script_name)

    return rt

def add_to_file(details, base_predictor, target_file):
    
    print ('Adding to target file:', target_file)
    
    data = []
    
    model_base_str = "'<<< READ MODEL HOLDER >>>'"
    transformer_base_str = "'<<< READ TRANSFORMER HOLDER >>>'"
    
    with open(base_predictor) as f:
        for row in f.readlines():
            
            # 1. Get the model load, and add to get_model
            if model_base_str in row:
                new_row = row.replace(model_base_str, details['cls.model'])
                data.append(new_row)
        
            # 2. Get the transformer load, and add to get_transformer
            elif transformer_base_str in row:
                new_row = row.replace(transformer_base_str, details['cls.transformer'])
                data.append(new_row)
                
            else:
                data.append(row)
            
    with open(target_file, 'w') as f:
        for row in data:
            f.writelines(row)
            
    return 

def create_deploy_endpoint(model_name, image_name):
    role = sagemaker.get_execution_role()
    
    client = boto3.client('sagemaker')

    model_dict = client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            'Image': image_name
        },
        ExecutionRoleArn=role)

    endpoint_config_name = '{}-config'.format(model_name)

    response = client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {   'VariantName': 'v',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.m4.xlarge'
            }])

    endpoint_name = '{}-endpoint'.format(model_name)
    
    print ('Deploying your endpoint:', endpoint_name)

    response = client.create_endpoint(
        EndpointName= endpoint_name,
        EndpointConfigName=endpoint_config_name)
    
    return



def add_to_base_container(source_file):
    
    base_file = 'Dockerfile_Base'
    
    target_file = 'container/Dockerfile'
    
    source_data = []
    
    print ('Parsing your file', source_file)
    
    with open(source_file) as f:
        for row in f.readlines():
            if len(row) >= 1:
                source_data.append(row)
        
    with open(base_file) as f:
        base_data = []
        for row in f.readlines():
            
            if 'py==' in row:
                splits = row.strip().split()
                for idx, each in enumerate(splits):
                    if '==' in each:
                        add_here = idx
                splits.insert(add_here+1, source_data[0])
                new_row = ' '.join(splits)
                base_data.append(new_row)
            else:
                base_data.append(row)
                
    with open(target_file, 'w') as f:
        for row in base_data:
            f.writelines(row)
            
    print ('Created your target file', target_file)

    return