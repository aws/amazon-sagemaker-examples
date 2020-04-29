# NOTE: should run in conda tensorflow_p36 (TF1) env.
import tensorflow as tf
import os
import sys
import shutil
import argparse

def split_tfrecord(data_dir, tfrecord_path, split_size):
    with tf.Graph().as_default(), tf.Session() as sess:
        ds = tf.data.TFRecordDataset(tfrecord_path).batch(split_size)
        batch = ds.make_one_shot_iterator().get_next()

        part_num = 0
        while True:
            try:
                records = sess.run(batch)
                #part_path = tfrecord_path + '.{:03d}'.format(part_num)

                if not os.path.exists(data_dir+f"/train/{part_num}"):
                    os.makedirs(data_dir+f"/train/{part_num}")
                output_file = os.path.join(data_dir, f"train/{part_num}/train_{part_num}.tfrecords")                
                print('Generating %s' % output_file)
                
                with tf.io.TFRecordWriter(output_file) as writer:
                    for record in records:
                        writer.write(record)
                part_num += 1
            except tf.errors.OutOfRangeError: break

def do_shard(data_dir, num_shards):
    tf.compat.v1.enable_eager_execution()
    
    input_file = os.path.join(data_dir, "train/train.tfrecords")
    raw_dataset = tf.data.TFRecordDataset(input_file)
    num_total_records = sum(1 for _ in raw_dataset)
    
    if num_total_records % num_shards != 0:
        print('Error: Number of total tfrecords ({}) are not a multiple of number of shards ({})!' % (num_total_records, num_shards))
        sys.exit(-1)
    else:
        size = num_total_records // num_shards
        split_tfrecord(data_dir, input_file, size)
        
if __name__ == '__main__':        
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        type=str,
        default='',
        help='Directory to download and extract CIFAR-10 to.')
    parser.add_argument(
        '--num-shards',
        type=int,
        default=1,
        help='Number of shards for Horovod.')

    args = parser.parse_args()
    do_shard(args.data_dir, args.num_shards)
