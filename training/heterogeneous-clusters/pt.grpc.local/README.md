### Local Training using gRPC Client-Server

The example here provides a conceptual idea of how to separate data preprocessing and training processes, and establish a gRPC client server communication between these components. We re-use the same client-server communication implementation in our Heterocluster SageMaker training example <PUT THE LINK when we move to aws-samples>. To demonstrate this we first run baseline testing where both the components (data preprocessing and training) run in a set of processes (see section A). In Section B, we split data processing run on gRPC Server set of processes whereas training run on a different set of processes.

Important:
Use a GPU based (preferably g4dn.xlarge) based SageMaker Notebook Instance, and use terminal window (File > New > Terminal). Tested on python 3.8 and Pytorch 1.10.
Note: This example does not support SageMaker "Studio" Notebook Instance. 

**Prerequsites**    
Pre-requisite 
Install dependent packages like pytorch, tensorboard, grpc. And, switch to working directory where all the scripts are stored.
```
pip install -r ~/SageMaker/hetro-training/pt.grpc.local/requirements.txt
cd ~/SageMaker/hetro-training/pt.grpc.local/
```
A. Baseline Testing
---

**Step 1**: Run basic mnist training script (no gRPC implementation). 
Modify the `main.py` to set:

`BATCH_SIZE = 8192`  
`ITERATIONS = 100` # No. of iterations in an epoc - must be multiple of 10s  
`DATALOADER_WORKERS = 2` #equals no. of CPUs of your local instance  

And, run the following commmand line:
```
python main.py
```

**Step 2**: Observe the `avg step time`. The steps/second starts printing on the console. And, stops after predefined iterations.
```    
sh-4.2$ python main.py
Training job started...
10: avg step time: 2.4617617287999565
20: avg step time: 2.4338118159999795
30: avg step time: 2.4230862849000006
40: avg step time: 2.42389511962499
50: avg step time: 2.4670148023599903
60: avg step time: 2.494912311566668
70: avg step time: 2.5143303409714335
80: avg step time: 2.530583710625001
90: avg step time: 2.5414934969444403
100: avg step time: 2.5489751799700024
Training completed!
```    
    
    
B. Split data pre-processing and training testing using gRPC Client-Server inter-process communication
---
In this example, we are decoupling the data pre-processing component of our training job, and the deep neural network (DNN) code. This way the data processing can run on CPU instance, and DNN on GPU instance. Here by introducing concept of heterogenous instances, but demonstrated both these processes running locally. The inter-process communucation is implemented by gRPC Client-Server communication. 

**Step 1**: Run gRPC Server in a new terminal session. The set of processes wait for client to request data. On request, it read the data from `data` folder, preprocess the data, and send it to the client for training. 
     
```
python main_grpc_server.py --batch-size 8192 --num-data-workers 4 --iterations 100 --grpc-workers 2
```
where,  
`batch-size` any integer  
`num-workers` based on no. of cpu per of your data pre-processing instance   
`iterations` no. of iterations in an epoch - must be multiple of 10s`   
`grpc-workers` no. of workers fetching the pre-processed data to DNN process (gRPC client)   

**Step 2**: Run gRPC Client in a new terminal session (File > New > Terminal) . The set of process spawend by this script fetches pre-processed data from server and runs training. Make sure you change your working directory to where code exist (..\..\pt.grpc.local). 
```
cd ~/SageMaker/hetro-training/pt.grpc.local/
python main_grpc_client.py --batch-size 8192 --num-dnn-workers 2 --iterations 100 --model-dir ./
```
where,   
`batch-size` any integer, must match to the size mentioned in the gRPC server process launch   
`num-workers` no. of dataloader workers, it is based on no. of cpu of the dnn instance   
`iterations` no. of iterations in an epoch - must be multiple of 10s, must match to the no. mentioned in the gRPC server process launch   
`model-dir` location of the model to be saved   

**Step 3**: Observe the `avg step time`. The steps/second starts printing to the console. And, stops after predefined iterations.
```    
sh-4.2$ python main_grpc_client.py 
Training job started...
10: avg step time: 0.43338242229997376
20: avg step time: 0.3908786807500064
30: avg step time: 0.3767881167999955
40: avg step time: 0.3698345946000018
50: avg step time: 0.3657775266399949
60: avg step time: 0.5407461890666657
70: avg step time: 0.6490721581571441
80: avg step time: 0.7544923407874989
90: avg step time: 0.8115056776666633
100: avg step time: 0.8806567812599997
Saving the model
Training job completed!
Shutting down data service via port 16000
``` 
    
**Step 4**: Optionally, in a new terminal window, you can validate whether the gRPC client-server communication is taking place. 

```
sh-4.2$ netstat -an | grep 6000
tcp6       0      0 :::6000                 :::*                    LISTEN     
tcp6       0      0 ::1:47888               ::1:6000                ESTABLISHED
tcp6       0      0 ::1:47890               ::1:6000                ESTABLISHED
tcp6       0      0 ::1:6000                ::1:47890               ESTABLISHED
tcp6       0      0 ::1:6000                ::1:47888               ESTABLISHED
```
C. Conclusion
---
In this example, we demonstrated concepts behind Heterogeneous cluster training. First, we ran a simple all-in one training script,(contains both data preprocessing and deep neural network(DNN) components), then we separated data pre-processing and dnn components to run as two different set of processes. We expand on these concepts in our next example [**PyTorch with gRPC distributed dataloader Heterogeneous Clusters training job example**](../pt.grpc.sagemaker/hetero-pytorch-mnist.ipynb). 