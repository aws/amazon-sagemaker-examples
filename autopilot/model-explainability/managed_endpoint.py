import boto3
region = boto3.Session().region_name

sm = boto3.Session().client(service_name='sagemaker',region_name=region)


class ManagedEndpoint:
    def __init__(self, ep_name, auto_delete=False):
        self.name = ep_name
        self.auto_delete = auto_delete
        
    def __enter__(self):
        endpoint_description = sm.describe_endpoint(EndpointName=self.name)
        if endpoint_description['EndpointStatus'] == 'InService':
            self.in_service = True        

    def __exit__(self, type, value, traceback):
        if self.in_service and self.auto_delete:
            print("Deleting the endpoint: {}".format(self.name))            
            sm.delete_endpoint(EndpointName=self.name)
            sm.get_waiter('endpoint_deleted').wait(EndpointName=self.name)
            self.in_service = False