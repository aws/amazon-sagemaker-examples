#!/usr/bin/env bash

if [ $1 = train ]
then
    echo "Container mode: train"
    export model_path="/opt/ml/model"
    export training_dir="/workspace/ngc_sagemaker_training"
    export pretrained_modeldir=$training_dir/pretrained_models
    export finetuned_modeldir=$training_dir/finetuned_modeldir
    mkdir -p $training_dir
    mkdir -p $pretrained_modeldir

    # download model from ngc
    cd $pretrained_modeldir
    wget https://api.ngc.nvidia.com/v2/models/nvidia/bert_tf_pretraining_lamb_16n/versions/1/files/model.ckpt-1564.meta
    wget https://api.ngc.nvidia.com/v2/models/nvidia/bert_tf_pretraining_lamb_16n/versions/1/files/model.ckpt-1564.index
    wget https://api.ngc.nvidia.com/v2/models/nvidia/bert_tf_pretraining_lamb_16n/versions/1/files/model.ckpt-1564.data-00000-of-00001
    wget https://api.ngc.nvidia.com/v2/models/nvidia/bert_tf_pretraining_lamb_16n/versions/1/files/bert_config.json
    wget https://api.ngc.nvidia.com/v2/models/nvidia/bert_tf_pretraining_lamb_16n/versions/1/files/vocab.txt

    # this would be the model script
    cd $training_dir
    git clone https://github.com/NVIDIA/DeepLearningExamples.git
    export bert_dir=$training_dir/DeepLearningExamples/TensorFlow/LanguageModeling/BERT
    export BERT_PREP_WORKING_DIR=$bert_dir/data

    # bertPrep.py from github imports PubMedTextFormatting - a package to be additionally installed that we don't need. So we copy our verison into the container.
    cp /opt/aws_sagemaker/infrastructure/bertPrep.py $BERT_PREP_WORKING_DIR
    # download data
    python3 $BERT_PREP_WORKING_DIR/bertPrep.py --action download --dataset squad


    # we want to add more flexibility to run_squad.sh
    cp /opt/aws_sagemaker/infrastructure/run_bert_squad.sh $bert_dir/scripts
    cd $bert_dir
    # get number of V100 gpus
    export num_V100=$(nvidia-smi | grep V100 |wc -l)
    # run fine-tuning
    # adjust the 0.2 below to run full fine-tuning, such as to 1.5
    bash scripts/run_bert_squad.sh 5 5e-6 fp16 true $num_V100 384 128 large 1.1 $pretrained_modeldir/model.ckpt-1564 0.2  $finetuned_modeldir true true
    
    # we have our fine-tuned model in $finetuned_modeldir, now convert to TRT
    cd $bert_dir/trt 
    cp /opt/aws_sagemaker/infrastructure/get_tf_model.py $bert_dir/trt 
    export latest_model_name=$(python get_tf_model.py $finetuned_modeldir/model.ckpt-*.meta)
    python builder.py -m $latest_model_name -o $model_path/bert_large_384.engine -b 1 -s 384 --fp16 -c $pretrained_modeldir
    
    # copy the vocab file and bert config file into /opt/ml/model
    cp $pretrained_modeldir/vocab.txt  $model_path
    cp $pretrained_modeldir/bert_config.json $model_path

elif [ $1 = serve ]
then
    echo "Container mode: serve"
    export model_path="/opt/ml/model"
    python3 /opt/aws_sagemaker/infrastructure/serve.py
else
    echo "Container mode not recognized. Should be: train or serve"
fi