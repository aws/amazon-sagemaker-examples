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

The next step is to deploy [game-server autopilot client](../ap_client)
