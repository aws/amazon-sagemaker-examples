## setup
Open bash and navigate to `./tf.data.service.sagemaker`

Set your SageMaker IAM role as an environment varaible. For example:
```
export SAGEMAKER_ROLE="arn:aws:iam::1234567890123:role/service-role/AmazonSageMaker-ExecutionRole-20171221T130536"
```

Start a homogeneous training job
```
python '/Users/gili/dev/hetro-training/tf.data.service.sagemaker/start_job.py' --tf_data_mode local --is_cloud_job --no-is_hetero --num_of_data_workers 0 --num_of_data_instances 0 --batch_size 1024
```

Start a heterogeneous training job
```
python '/Users/gili/dev/hetro-training/tf.data.service.sagemaker/start_job.py' --tf_data_mode service --is_cloud_job  --is_hetero  --num_of_data_workers 2 --num_of_data_instances 2 --batch_size 1024
```