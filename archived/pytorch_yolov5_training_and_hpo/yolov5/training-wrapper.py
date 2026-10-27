#!/usr/bin/env python3

import os
import sys
import json
import argparse
import shutil
import shlex
import subprocess
import yaml 
import boto3
import botocore

DATACFGFILE = '/yolov5/datacfg.yaml'
SMHYPFILE = '/yolov5/sagemaker-hyps.yaml'
DEFYOLOHYPFILE = '/yolov5/data/hyps/hyp.scratch-low.yaml'

def buildDataConfig(trainingdata,validationdata):
    ## dictfile = {"train": trainingdata + "/images", "val": validationdata + "/images", "nc": 1, "names": ["licence"]}
    ## Read from the JSON /yolov5/classenum.json file
    with open('/yolov5/classenum.json', 'r') as classfile:
        clases = json.load(classfile)
    num_classes = len(clases)
    classlist = [cl['name'].split('.')[0] for cl in clases]
    dictfile = {"path": "./custom_data", "train": "images/train", "val": "images/val", "nc": num_classes, "names": classlist}
    with open(DATACFGFILE,'w') as fil:
        docs = yaml.dump(dictfile,fil)
    return DATACFGFILE

def createHypsFile(opts):
    ## Create a new file that we can use
    if os.path.exists(SMHYPFILE):
        os.remove(SMHYPFILE)
    try:
        hypfile = open(SMHYPFILE,'w') # New file
    except OSError:
        print("Could not open file {} for training initialization".format(SMHYPFILE)) 
        sys.exit(44)
    try:
        args = yaml.dump(opts,hypfile)
    except:    
        print("** TRAINING FAILED TO INITIALIZE **")
        hypfile.close()
        sys.exit(45) 
    hypfile.close()
    return " --hyp " + SMHYPFILE

def buildCMD(arguments):
    # Initialize the non-hyps hyper-parameters
    non_hyps = ['batchsize','freeze','epochs','patience','name']
    opts = dict()
    ## Our default hyperparameter values are sourced from 
    ## yolov5/data/hyps/hyp.scratch-low.yaml
    ## We then override what we want changed.
    try:
        defhyps = open(DEFYOLOHYPFILE)
    except OSError:
        print("Could not open file {} for training initialization".format(DEFYOLOHYPFILE))
    try:
        params = yaml.full_load(defhyps)
    except:
        print("** TRAINING FAILED TO INITIALIZE **")
        defhyps.close()
        sys.exit(43)
    defhyps.close()
    # Set up default hyper-parameter values
    for item,value in params.items():
        opts[item] = value
    initialstr = list()
    initialstr.append("/opt/conda/bin/python3.8 /yolov5/train.py --project /opt/ml/model --cache ")

    argls = (arguments.__dict__).items()    
    
    for arg in argls:
        if arg[1] is None or arg[0] in ['train','val']:
            continue
        elif arg[1] is not None and arg[0] in non_hyps:
            if arg[0] == 'batchsize':
                initialstr.append(" --batch " + str(arg[1]))
            else:
                initialstr.append(" --" + str(arg[0]) + " " + str(arg[1]))
        else:
            opts[str(arg[0])] = arg[1]
    initialstr.append(createHypsFile(opts))

    return ''.join(initialstr)

def separateimages(prefx,chan):
    '''Here prefx is the full path to the parent of train or val'''
    pngs = [f for f in os.listdir(prefx + '/' + chan) if f.endswith('JPG')]
    for png in pngs:
        shutil.copyfile(prefx + '/' + chan + '/' + png,'/yolov5/custom_data' + '/images/' + chan + '/' + png)
    txts = [f for f in os.listdir(prefx + '/' + chan) if f.endswith('txt')] 
    for t in txts:
        print("Copying label file : {}".format(t))
        shutil.copyfile(prefx + '/' + chan + '/' + t,'/yolov5/custom_data' + '/labels/' + chan + '/' + t)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # sagemaker-containers passes hyperparameters as arguments
    parser.add_argument("--img-size", type=int, default=640,required=False)
    parser.add_argument("--batchsize", type=int,default=16,required=True)
    parser.add_argument("--epochs", type=int,default=500,required=True)
    parser.add_argument("--weights", type=str,required=False)
    parser.add_argument("--freeze", type=int,required=False)
    parser.add_argument("--patience",type=int,required=False)
    parser.add_argument("--hyp",type=str,required=False)
    parser.add_argument("--classes",type=str,required=True)
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--val', type=str, default=os.environ['SM_CHANNEL_VAL'])
    parser.add_argument('--name', type=str,required=False)

    # YOLOv5 HPO arguments, apart from batch and epochs etc. considered above
    # We will use the data/hyps/hyp.scratch.p5.yaml as the source of all default values.
    parser.add_argument("--lr0",type=float,required=False)
    parser.add_argument("--lrf",type=float,required=False)
    parser.add_argument("--momentum",type=float,required=False)
    parser.add_argument("--weight_decay",type=float,required=False)
    parser.add_argument("--warmup_epochs",type=int,required=False)
    parser.add_argument("--warmup_momentum",type=float,required=False)
    parser.add_argument("--warmup_bias_lr",type=float,required=False)
    parser.add_argument("--box",type=float,required=False)
    parser.add_argument("--cls",type=float,required=False)
    parser.add_argument("--cls_pw",type=float,required=False)
    parser.add_argument("--obj",type=float,required=False)
    parser.add_argument("--obj_pw",type=float,required=False)
    parser.add_argument("--iou_t",type=float,required=False)
    parser.add_argument("--anchor_t",type=float,required=False)
    parser.add_argument("--fl_gamma",type=float,required=False)
    parser.add_argument("--hsv_h",type=float,required=False)
    parser.add_argument("--hsv_s",type=float,required=False)
    parser.add_argument("--hsv_v",type=float,required=False)
    parser.add_argument("--degrees",type=float,required=False)
    parser.add_argument("--translate",type=float,required=False)
    parser.add_argument("--scale",type=float,required=False)
    parser.add_argument("--shear",type=float,required=False)
    parser.add_argument("--perspective",type=float,required=False)
    parser.add_argument("--flipud",type=float,required=False)
    parser.add_argument("--fliplr",type=float,required=False)
    parser.add_argument("--mosaic",type=float,required=False)
    parser.add_argument("--mixup",type=float,required=False)
    parser.add_argument("--copy_paste",type=float,required=False)

    args = parser.parse_args()
    ###
    if not os.path.exists('/yolov5/custom_data'):
        os.makedirs('/yolov5/custom_data')
        os.makedirs('/yolov5/custom_data/images')
        os.makedirs('/yolov5/custom_data/labels')
        os.makedirs('/yolov5/custom_data/images/train') ## Training images
        os.makedirs('/yolov5/custom_data/images/val') ## Validation images
        os.makedirs('/yolov5/custom_data/labels/train') ## Training labels
        os.makedirs('/yolov5/custom_data/labels/val') ## Validation labels
    ###
    ## This covers two channels - train, and val
    separateimages('/opt/ml/input/data','train')
    separateimages('/opt/ml/input/data','val')
    os.chdir('/yolov5') ## New directory
    cmd = buildCMD(args)
    ## get classes from S3. We need these to build out the cfg file
    if args.classes:
        s3 = boto3.resource('s3')
        bucketname = (args.classes).split('/')[2]
        key = (args.classes).split(bucketname)[-1].lstrip('/')
        try:
            s3.Bucket(bucketname).download_file(key, '/yolov5/' + key.split('/')[-1])
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise
    ## Standard Amazon SageMaker paths, you could add a test path as well
    cmd += " --data " + buildDataConfig('/opt/ml/input/data/train','/opt/ml/input/data/val')

    ## get weights from S3, if we need it, else we just leave it to the YOLOv5 train.py
    ## to get the weights from the official github repo.
    if args.weights:
        s3 = boto3.resource('s3')
        bucketname = (args.weights).split('/')[2]
        key = (args.weights).split(bucketname)[-1]
        cmd += " --weights " + "/yolov5/" + key.split('/')[-1]
        try:
            s3.Bucket(bucketname).download_file(key[1:], '/yolov5/' + key.split('/')[-1])
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise
    print("**Data Configuration**:")
    with open(DATACFGFILE) as f:
        print(f.read())
    print("**Hyperparameters Configuration**:")
    with open(SMHYPFILE) as f:
        print(f.read())
    print("**Executing** : {}".format(cmd))
    try:
        subprocess.run(shlex.split(cmd),check=True,encoding="utf-8")
    except subprocess.CalledProcessError as e:
        print("###*** TRAINING FAILED ***###")
        print("Returned code : {}".format(e.returncode))
        sys.exit(45)
