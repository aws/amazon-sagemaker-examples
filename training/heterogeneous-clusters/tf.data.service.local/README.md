# tf.data.service - running locally
This example runs the tf.data.service locally on your machine (not on SageMaker). This will help you understand how tf.data.service dispatcher, worker and client work.

## setup
Open bash and navigate to `./tf.data.service.local`
```bash
cd ./tf.data.service.local
```

Install requirements:
```bash
pip install -r requirements.txt
```
## Running without tf.data.service
First lets run a single training process that handles both data augmentation and NN optimization:
```bash
python3 ./train.py --mode local --model-dir /tmp
```
Expected output:
```
Running in local mode
1/1 [==============================] - 33s 33s/step - loss: 3.9110
```
## Running with tf.data.service
Now let's run the same trianing job in two process utilizing tf.data.service.  

We first run the tf.data.service dispatcher and worker processes which will handle some of the heavy data augmentation tasks:
```bash
python3 ./run-dispatcher-and-worker.py
```
Expected output:
```
2022-06-28 18:10:50.337939: I tensorflow/core/data/service/server_lib.cc:64] Started tf.data DispatchServer running at 0.0.0.0:6000
2022-06-28 18:10:50.350973: I tensorflow/core/data/service/worker_impl.cc:148] Worker registered with dispatcher running at localhost:6000
2022-06-28 18:10:50.351535: I tensorflow/core/data/service/server_lib.cc:64] Started tf.data WorkerServer running at 0.0.0.0:6001
```
Now let's launch the NN training script which will connect to the dispatcher to consume its data.source
```bash
python3 ./train.py --mode service --model-dir /tmp
```
Expected output:
```
Running in service mode
1/1 [==============================] - 34s 34s/step - loss: 3.9806
```
Done.  
Next see [**TensorFlow's tf.data.service with Amazon SageMaker Training Heterogeneous Clusters**](../tf.data.service.sagemaker/hetero-tensorflow-restnet50.ipynb)