import os
import re
import math
import json
import boto3
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from time import gmtime, strftime

pricing_client = boto3.client('pricing')


def run_load_tests(api_url, endpoints_list):
    """
    Runs a locust load test for a given list of endpoints.
    
    Given a set of SageMaker endpoints and an API Gateway URL, this
    function runs the run_locust.sh file for each endpoint. For easy 
    reference, the result files are organized into a folder with the 
    endpoint name and time prefix.
    
    Inputs:
    api_url: API Gateway URL for load testing.
    endpoints_list: List of SageMaker endpoints to run the load tests on.
    
    Output:
    Folder name where the load test results are saved.
    """ 
    
    time_stamp = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    # Iterate over the different endpoints
    for endpoint in endpoints_list:
        
        print(f"\nLoad testing {endpoint}...")
        subprocess.run(["bash", "run_locust.sh", f"{endpoint}", f"{api_url}"], stdout=subprocess.PIPE)
        
        print(f"Organizing {endpoint} result files...")
        
        prefix = f"test_{endpoint}"
        os.makedirs(f"results-{time_stamp}/{prefix}")
        
        files = [f for f in os.listdir() if prefix in f]
        for file in files:
            os.rename(file, f"results-{time_stamp}/{prefix}/{file}")
            
    print("Load testing complete!")
            
    return f"results-{time_stamp}"   



def get_pricing(instance_type, duration='hour'):
    """
    Return current SageMaker pricing for a given instance.
    
    Using the AWS PriceList API, this function returns the current 
    pricing for a given instance type. This function is called to plot
    the performance vs. price plots.
    
    Inputs:
    instance_type: SageMaker instance type, e.g., ml.m5.xlarge
    duration: time for pricing - default at hour. Valid options are 
        hour, day or month. 
        
    Output:
    Price of the instance in USD.
    """
    
    flag = False
    
    # Validate 'duration'
    if duration not in ['hour', 'day', 'month']:
        print("Duration should be one of hour, day or month. Exiting..")
        return 0
    
    # Get pricing
    resp = pricing_client.get_products(
        ServiceCode='AmazonSageMaker',
        Filters=[
            {
                'Type': 'TERM_MATCH',
                'Field': 'instanceType',
                'Value': f"{instance_type}-Hosting"
            }
        ]
    )
    for price_list in resp['PriceList']:
        # Get pricing for us-east-1
        if json.loads(price_list)['product']['attributes']['location'] == "US East (N. Virginia)":
            x1 = json.loads(price_list)['terms']['OnDemand']
            x1_key1 = list(x1.keys())[0]
            x2 = x1[x1_key1]['priceDimensions']
            x2_key2 = list(x2.keys())[0]
            x3 = x2[x2_key2]['pricePerUnit']['USD']
            flag = True
    
    if flag == False:
        print("Could not find instance in us-east-1. Exiting..")
        return 0
    
    if duration=='hour':
        return round(float(x3), 3)
    elif duration=='day':
        return round(float(x3), 3) * 24
    elif duration=='month':
        return round(float(x3), 3) * 24 * 30
    
    
def generate_plots(endpoints, endpoints_dict, results_folder, sep_cpu_gpu=False):
    """
    Generate plots comparing the performance vs price
    
    This function calls the get_pricing function to get instance prices,
    and plots them against the performance of each endpoint. 
    
    Inputs:
    endpoints: list of endpoint names
    endpoints_dict: list of endpoint names with instance counts
    results_folder: path where load test results are saved
    sep_cpu_gpu: bool whether to plot CPU and GPU results in separate plots.
        defaults to False
        
    Output:
    Matplotlib plot showing performance against price.
    """
    prices = {}

    for item in endpoints_dict:
        instance = item['instance_type']
        count = item['instance_count']
        cost = get_pricing(instance)
        prices.update({
            f"{instance}.x{count}" : cost * count
        })

    # Get max requests for all instance types
    max_requests = {}

    for ep in endpoints:
        prefix = f"test_{ep}"
        df = pd.read_csv(f"{results_folder}/{prefix}/{prefix}_stats_history.csv")
        fail_at_1 = df.tail(1)['Requests/s'].values[0]
        max_requests.update({
            ep.split("-", 1)[1].replace("-", "."): fail_at_1
        })
     

    results = pd.DataFrame([prices, max_requests]).T
    results.columns = ['Price per Hour', 'Max Requests per Second']
    # Round down requests per second to integer
    results['Max Requests per Second'] = results['Max Requests per Second'].apply(
        lambda x: math.floor(x)
    )
    
    # get cpu-gpu flag
    results['type'] = results.index.str.split('.')
    results['type'] = results['type'].apply(lambda x: x[1])
    results['gpu_flag'] = results['type'].apply(lambda x: 1 if x.startswith(('p', 'g', 'e', 'i')) else 0)
    results.drop(['type'], axis=1, inplace=True)

    if sep_cpu_gpu:
        cpu_df = results[results['gpu_flag'] == 0]
        gpu_df = results[results['gpu_flag'] == 1]
        fig, ax = plt.subplots(1,2, figsize=(15,6))
        
        # plot cpu instances
        ax[0].scatter(cpu_df['Price per Hour'], cpu_df['Max Requests per Second'])
        ax[0].title.set_text('CPU Instances')
        ax[0].set_xlabel('Instance Price per Hour')
        ax[0].set_ylabel('Max. Requests per Second')
        for i, row in cpu_df.iterrows():
            ax[0].annotate(i, (row['Price per Hour'], row['Max Requests per Second']))
        ax[0].grid('True')
        
        # plot gpu instances
        ax[1].scatter(gpu_df['Price per Hour'], gpu_df['Max Requests per Second'])
        ax[1].title.set_text('GPU Instances')
        ax[1].set_xlabel('Instance Price per Hour')
        ax[1].set_ylabel('Max. Requests per Second')
        for i, row in gpu_df.iterrows():
            ax[1].annotate(i, (row['Price per Hour'], row['Max Requests per Second']))
        ax[1].grid('True')
        
        fig.suptitle("Pricing vs Performance Plot")
        plt.show()
        
    else:
        plt.figure(figsize=(12,7))

        for i in results.values:
            plt.scatter(i[0], i[1])

        plt.title('Pricing vs Performance Plot', fontsize=15)
        plt.xlabel('Instance Price per Hour', fontsize=12)
        plt.ylabel('Max. Requests per Second', fontsize=12)
        plt.legend(results.index)
        plt.grid('True')
        plt.show()
    
    return results.drop(['gpu_flag'], axis=1)


def generate_latency_plot(endpoints, results_folder):
    """
    Generate latency plots for a given list of endpoints
    
    Given a list of endpoints, this function plots the minimum, 
    maximum and average latency in a box plot.
    
    Inputs:
    endpoints: list of endpoint names
    results_folder: path where load test results are saved
    
    Output:
    Boxplot of latencies per instance type
    """
    latency_dict= {}

    # for each endpoint get latency from the load test results
    for ep in endpoints:
        prefix = f"test_{ep}"
        df = pd.read_csv(f"{results_folder}/{prefix}/{prefix}_stats.csv")

        min_latency = round(df.tail(1)['Min Response Time'].values[0] / 1000, 1)
        avg_latency = round(df.tail(1)['Average Response Time'].values[0] / 1000, 1)
        max_latency = round(df.tail(1)['Max Response Time'].values[0] / 1000, 1)

        latency_dict.update({
            ep.split("-", 1)[1].replace("-", "."): [min_latency,
                                                    avg_latency,
                                                    max_latency]
        })

    # generate a data frame of the results and plot a box plot
    results = pd.DataFrame(latency_dict)
    res_plot = results.plot(legend=True, figsize=(12, 7), kind='box',
                title="Request Latency Across Instance Types",
                xlabel="Instance Type",
                ylabel="Latency in Seconds")

    
def get_min_max_instances(results_df, min_requests, max_requests):
    """
    Calculates recommendations for autoscaling
    
    Based on the maximum requests handled by each endpoint, this function
    calculates and returns the optimal instance count and type for an
    autoscaling configuration. 
    
    Inputs:
    results_df: pandas data frame with instance types and their maximum rps
    min_requests: minimum number of requests per second required for the application
    max_requests: maximum number of requests per second required for the application
    
    Output:
    Recommended instance type and count for optimal costs
    
    """
    if max_requests < min_requests:
        print("Minimum requests should be less than or equal to the maximum number of requests per second. Exiting..")
        return
    
    # calculate min and max number of instance required for each instance type
    # to serve the min and max rps, and calculate the corresponding prices
    results_df = results_df.copy(deep=True)
    results_df['Min Instances'] = results_df['Max Requests per Second'].apply(lambda x: round(min_requests / x))
    results_df['Pricing'] = results_df.apply(lambda x: x['Price per Hour'] * x['Min Instances'], axis=1)
    results_df = results_df.sort_values(['Pricing'])
    results_df = results_df[results_df['Min Instances'] > 0]
    results_df['Max Instances'] = results_df['Max Requests per Second'].apply(lambda x: round(max_requests / x))

    # recommended type is the top row of the sorted data frame
    recommended_type = results_df.head(1).index.values[0]
    recommended_type = re.sub(r'.x[0-9]', '', recommended_type)
    recommended_min = results_df.head(1)['Min Instances'].values[0]
    recommended_max = results_df.head(1)['Max Instances'].values[0]
    
    recommended_dict = [    
        {"instance_type": recommended_type,  "instance_count": int(recommended_min)},
        {"instance_type": recommended_type, "instance_count": int(recommended_max)}
    ]
    return recommended_dict