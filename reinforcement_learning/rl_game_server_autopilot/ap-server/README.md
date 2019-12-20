## Deploy game-server autopilot server
Gameserver autopilot includes a server and client components. The server comprises of the endpoint deployed via SageMaker Inference, DynamoDB table to persist history of inferences, and an API gateway and Lambda function (http://tbd/) to simplify SageMaker runtime semantics. The [autopilot client](../ap-client) is deployed as a Kubernetes Pod that queries the autopilot server for the number of game-server needed to be launched for the next 10 minutes.

This step can be executed after the model training complete and the model is deployed and availiable through a SageMaker Inference Endpoint. i.e., after completing [Model Deployment](https://github.com/yahavb/amazon-sagemaker-examples/tree/master/reinforcement_learning/rl_game_server_autopilot/sagemaker/rl_gamerserver_ray.ipynb#Model-deployment). The game-server autopilot server runs a lambda function behind API gateway. We use [Chalice](https://chalice.readthedocs.io/en/latest/) to deploy and intergate the lambda function and the API gateway endpoint. Autopilot server also masks the access to the EKS control-plane to determine the current demand for game-servers. 

Before deploying the lambda function that powers autopilot server, populate the variable `endpoint_names` that denotes the model inference endpoint in [app.py](app.py). Also populate the `gs_inventory_url` variable with the EKS API Server. 

Deploy the autopilot server by executing:

```bash
cd ap-server/
chalice deploy
``` 
Chalice create a lambda function and a role it assumed upon a call to the server. The role should have `dynamodb:GetItem` and `dynamodb:PutItem` permissions to `table/latest_observations`. 

Test it by invoking the `/predict` method, e.g.,

```bash
[ap-server]$http https://o2p5pgyyk0.execute-api.us-west-2.amazonaws.com/api/predict/us-west-2
HTTP/1.1 200 OK
Connection: keep-alive
Content-Length: 322
Content-Type: application/json
Date: Fri, 20 Dec 2019 05:43:44 GMT
Via: 1.1 8275ae3e861a04a309ec8b466cdc4a26.cloudfront.net (CloudFront)
X-Amz-Cf-Id: 1sVARwsmZMr7THDFwkQOm-QPCQ3xy4qlFX44-WzEvYoBHL8ldDwD9A==
X-Amz-Cf-Pop: SEA19-C1
X-Amzn-Trace-Id: Root=1-5dfc5f8f-59f3527c67c84bd42559ac2c;Sampled=0
X-Cache: Miss from cloudfront
x-amz-apigw-id: E_PebEMrvHcFxyg=
x-amzn-RequestId: 0b5e6c94-2f8e-432f-97e5-fa589582e9ab

{
    "Prediction": {
        "action": "28.546",
        "curr_demand": "94.85064306952303",
        "is_false_positive": 1,
        "new_alloc": "94.85064306952303",
        "num_of_gameservers": 94.85064306952303,
        "observations": "[10.         10.         10.         10.         94.85064307]",
        "raw_results": {
            "predictions": [
                {
                    "actions": [
                        0.28546
                    ],
                    "logits": [
                        0.277642,
                        -5.3319
                    ]
                }
            ]
        }
    }
}
```


## Deploy game-server autopilot client
Now that we have the game server running, we can schedule the autopilot client to autoscale based on predictions. It uses a trained model that predicts the number of game-servers needed. In this workshop, we are going to focus on the client side only. The client is backed by a model that learns usage patterns and adopts the predictions based on emerging game-server allocation in a specific cluster. The Autopilot client sets the necessary number of game-servers. If there is a need for more EC2 instances, there will be game-server jobs that are pending. That will trigger the cluster_autoscaler that we deployed in previous step to add more EC2 instances, making space for more pods.

   To deploy autopilot execute:
   1. Build and deploy the image to ECR.
   ```
   cd autopilot-image
   ./build.sh
   cd ../..
   ```

   Then update the [autopilot-client-deploy.yaml](specs/autopilot-client-deploy.yaml) with the image name, and deploy it
   ```
   kubectl apply -f specs/autopilot-client-deploy.yaml
   ```

   After the pod is scheduled check its stdour/err by executing:

   ```
   kubectl logs `kubectl get po | grep autopilot | awk '{print $1}'`
   ```

   After few minutes, we can start seeing metrics populated in cloudwatch.
   Using CloudWatch console, discover the `multiplayersample` CloudWatch namespace under Custom Namespaces. There are five metrics that help us to assess the system health.
   * `num_of_gs` - Describes the predicated number of game-server that was set on the cluster.
   * `current_gs_demand` - Describes the current demand for game-servers by players.
   * `num_of_nodes` - Describes the number of EC2 instances allocated.
   * `false-positive` - Counter of cases where the predictions `num_of_gs` was smaller than `current_gs_demand` and could cause live session interruption.
   1. Create a line graph that includes `num_of_gs` and `current_gs_demand` to assess the prediction quality. Set the metric data aggregation to 5min **(Statistics=Average)**
   2. Create a line graph that includes `num_of_gs` and `num_of_nodes` to assess the correlation between game-server allocation and EC2 instances allocation. Set the metric data aggregation to 5min **(Statistics=Average)**
   3. Create a line graph that aggregates the number of **false positives** by the autopilot model. Set the metric aggregation to 5min **(Statistics=Sum)**
   Resulted ![CloudWatch Dashboard](ap-cloudwatch.png)

