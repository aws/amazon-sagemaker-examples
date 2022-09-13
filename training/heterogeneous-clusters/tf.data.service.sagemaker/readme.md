## setup
Open bash and navigate to `./tf.data.service.sagemaker`

Let's generate some data to train over:
```bash
python3 ./generate_cifar10_tfrecords.py --data-dir ./data
rm -rf /tmp/data.old && mv data data.old && mkdir data && cp data.old/train/train.tfrecords ./data/ && mv data.old /tmp
```
This use SageMaker default S3 bucket to store the dataset, you may choose a different bucket
```bash
export S3_BUCKET_DATASET=$(python3 -c "import sagemaker; print(sagemaker.Session().default_bucket())")
```
Copy the dataset to the S3 bucket:
```
aws s3 sync ./data/ s3://${S3_BUCKET_DATASET}/cifar10-tfrecord/
```
If you are not using the default bucket, then edit `start_job.py` and set `S3_BUCKET_DATASET` to your prefered S3 bucket.

Set your SageMaker IAM role as an environment varaible. For example:
```
export SAGEMAKER_ROLE="arn:aws:iam::1234567890123:role/service-role/AmazonSageMaker-ExecutionRole-20171221T130536"
```

Start a homogenous training job
```
python '/Users/gili/dev/hetro-training/tf.data.service.sagemaker/start_job.py' --tf_data_mode local --is_cloud_job --no-is_hetero --num_of_data_workers 0 --num_of_data_instances 0 --batch_size 1024
```

Start a heterogenous training job
```
python '/Users/gili/dev/hetro-training/tf.data.service.sagemaker/start_job.py' --tf_data_mode service --is_cloud_job  --is_hetero  --num_of_data_workers 2 --num_of_data_instances 2 --batch_size 1024
```