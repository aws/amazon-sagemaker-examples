"""Train Faster-RCNN end to end."""
import argparse
import os

# disable autotune
os.environ[
    "MXNET_CUDNN_AUTOTUNE_DEFAULT"
] = "0"  ###Use autotone then allow mxnet to find params and later update.
os.environ["MXNET_GPU_MEM_POOL_TYPE"] = "Round"
os.environ["MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF"] = "26"
os.environ["MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD"] = "999"
os.environ["MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD"] = "25"
os.environ["MXNET_GPU_COPY_NTHREADS"] = "1"
os.environ["MXNET_OPTIMIZER_AGGREGATION_SIZE"] = "54"
os.environ["OMPI_MCA_btl_vader_single_copy_mechanism"] = "none"


os.environ["HOROVOD_GPU_ALLREDUCE"] = "NCCL"
os.environ["HOROVOD_FUSION_THRESHOLD"] = "134217728"
os.environ["HOROVOD_NUM_STREAMS"] = "2"
os.environ["MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD"] = "999"
os.environ["MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD"] = "25"
os.environ["OMP_NUM_THREADS"] = "2"

os.environ["HOROVOD_CYCLE_TIME"] = "0.1"
os.environ["HOROVOD_HIERARCHICAL_ALLREDUCE"] = "0"
os.environ["HOROVOD_CACHE_CAPACITY"] = "0"

os.environ["NCCL_MIN_NRINGS"] = "1"
os.environ["NCCL_TREE_THRESHOLD"] = "4294967296"
os.environ["NCCL_NSOCKS_PERTHREAD"] = "8"
os.environ["NCCL_SOCKET_NTHREADS"] = "2"
os.environ["NCCL_BUFFSIZE"] = "16777216"
os.environ["HOROVOD_NUM_NCCL_STREAMS"] = "2"

os.environ["NCCL_NET_GDR_READ"] = "1"
os.environ["HOROVOD_TWO_STAGE_LOOP"] = "1"
os.environ["HOROVOD_ALLREDUCE_MODE"] = "1"
os.environ["HOROVOD_FIXED_PAYLOAD"] = "161"
os.environ["HOROVOD_MPI_THREADS_DISABLE"] = "1"
os.environ["MXNET_USE_FUSION"] = "0"

import logging
import time

import gluoncv as gcv
import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet.contrib import amp

gcv.utils.check_version("0.7.0")
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.data.batchify import Append, FasterRCNNTrainBatchify, Tuple
from gluoncv.data.transforms.presets.rcnn import (
    FasterRCNNDefaultTrainTransform,
    FasterRCNNDefaultValTransform,
)
from gluoncv.model_zoo import get_model
from gluoncv.model_zoo.rcnn.faster_rcnn.data_parallel import ForwardBackwardTask
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.utils.metrics.rcnn import (
    RCNNAccMetric,
    RCNNL1LossMetric,
    RPNAccMetric,
    RPNL1LossMetric,
)
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.parallel import Parallel

try:
    import horovod.mxnet as hvd
except ImportError:
    hvd = None

try:
    from mpi4py import MPI
except ImportError:
    logging.info('mpi4py is not installed. Use "pip install --no-cache mpi4py" to install')
    MPI = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN network end to end.")
    parser.add_argument("--datasetloc", type=str, default="", help="where the dataset is located")
    parser.add_argument("--sm-save", type=str, default="", help="where to save models")
    parser.add_argument("--sm-output", type=str, default="", help="where to save data for sm")
    parser.add_argument(
        "--num-workers",
        "-j",
        dest="num_workers",
        type=int,
        default=4,
        help="Number of data workers, you can use larger "
        "number to accelerate data loading, if you CPU and GPUs "
        "are powerful.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Training mini-batch size.")
    parser.add_argument(
        "--gpus", type=str, default="0", help="Training with GPUs, you can specify 1,3 for example."
    )
    parser.add_argument("--epochs", type=str, default="", help="Training epochs.")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Resume from previously saved parameters if not None. "
        "For example, you can resume from ./mask_rcnn_xxx_0123.params",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="Starting epoch for resuming, default is 0 for new training."
        "You can specify it to 100 for example to start from 100 epoch.",
    )
    parser.add_argument(
        "--lr",
        type=str,
        default="",
        help="Learning rate, default is 0.01 for coco 8 gpus training.",
    )
    parser.add_argument(
        "--lr-decay", type=float, default=0.1, help="decay rate of learning rate. default is 0.1."
    )
    parser.add_argument(
        "--lr-decay-epoch",
        type=str,
        default="",
        help="epochs at which learning rate decays. default is 17,23 for coco.",
    )
    parser.add_argument(
        "--lr-warmup",
        type=str,
        default="",
        help="warmup iterations to adjust learning rate, default is 1000 for coco.",
    )
    parser.add_argument(
        "--lr-warmup-factor", type=float, default=1.0 / 3.0, help="warmup factor of base lr."
    )
    parser.add_argument("--clip-gradient", type=float, default=-1.0, help="gradient clipping.")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum, default is 0.9")
    parser.add_argument("--wd", type=str, default="", help="Weight decay, default is 1e-4 for coco")
    parser.add_argument(
        "--log-interval", type=int, default=100, help="Logging mini-batch interval. Default is 100."
    )
    parser.add_argument("--save-prefix", type=str, default="", help="Saving parameter prefix")
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1,
        help="Saving parameters epoch interval, best model will always be saved.",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=1,
        help="Epoch interval for validation, increase the number will reduce the "
        "training time if validation is slow.",
    )
    parser.add_argument("--seed", type=int, default=233, help="Random seed to be fixed.")
    parser.add_argument(
        "--verbose", type=str, default="false", help="Print helpful debugging info once set."
    )
    # Norm layer options
    parser.add_argument(
        "--norm-layer",
        type=str,
        default=None,
        help="Type of normalization layer to use. "
        "If set to None, backbone normalization layer will be fixed,"
        " and no normalization layer will be used. "
        "Currently supports 'bn', and None, default is None",
    )
    parser.add_argument(
        "--amp", type=str, default="false", help="Use MXNet AMP for mixed precision training."
    )
    parser.add_argument(
        "--horovod",
        type=str,
        default="false",
        help="Use MXNet Horovod for distributed training. Must be run with OpenMPI. "
        "--gpus is ignored when using --horovod.",
    )
    parser.add_argument(
        "--executor-threads",
        type=int,
        default=1,
        help="Number of threads for executor for scheduling ops. "
        "More threads may incur higher GPU memory footprint, "
        "but may speed up throughput. Note that when horovod is used, "
        "it is set to 1.",
    )
    parser.add_argument(
        "--kv-store",
        type=str,
        default="nccl",
        help="KV store options. local, device, nccl, dist_sync, dist_device_sync, "
        "dist_async are available.",
    )

    args = parser.parse_args()

    def str_2_bool(args, b=False):
        if args.lower() == "true":
            return True
        else:
            if b:
                return False
            return None

    args.verbose = str_2_bool(args.verbose)
    args.amp = str_2_bool(args.amp)
    args.horovod = str_2_bool(args.horovod)
    keys = list(os.environ.keys())
    args.sm_save = (
        os.path.join(os.environ["SM_MODEL_DIR"], args.sm_save)
        if "SM_MODEL_DIR" in keys
        else args.sm_save
    )
    args.num_workers = args.num_workers
    args.datasetloc = (
        os.environ["SM_CHANNEL_DATA"] if "SM_CHANNEL_DATA" in keys else args.datasetloc
    )
    args.gpus = int(os.environ["SM_NUM_GPUS"]) if "SM_NUM_GPUS" in keys else int(args.gpus)
    args.sm_output = (
        os.path.join(os.environ["SM_OUTPUT_DATA_DIR"], args.sm_output)
        if "SM_OUTPUT_DATA_DIR" in keys
        else args.sm_output
    )
    args.batch_size = int(args.batch_size)

    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)
    if args.horovod:
        if hvd is None:
            raise SystemExit("Horovod not found, please check if you installed it correctly.")
        hvd.init()

    args.epochs = int(args.epochs) if args.epochs else 26
    args.lr_decay_epoch = args.lr_decay_epoch if args.lr_decay_epoch else "8,11"
    args.lr = float(args.lr) if args.lr else 0.00125
    args.lr_warmup = args.lr_warmup if args.lr_warmup else 1000
    args.wd = float(args.wd) if args.wd else 1e-4

    def str_args2num_args(arguments, args_name, num_type):
        try:
            ret = [num_type(x) for x in arguments.split(",")]
            if len(ret) == 1:
                return ret[0]
            return ret
        except ValueError:
            raise ValueError("invalid value for", args_name, arguments)

    return args


def get_dataset(args):
    train_dataset = gdata.COCODetection(
        root=args.datasetloc, splits="instances_train2017", use_crowd=False, skip_empty=True
    )
    val_dataset = gdata.COCODetection(
        root=args.datasetloc, splits="instances_val2017", skip_empty=False
    )
    val_metric = COCODetectionMetric(val_dataset, args.save_prefix + "_eval")
    return train_dataset, val_dataset, val_metric


def get_dataloader(
    net, train_dataset, val_dataset, train_transform, val_transform, batch_size, num_shards, args
):
    """Get dataloader."""
    train_bfn = FasterRCNNTrainBatchify(net, num_shards)
    if hasattr(train_dataset, "get_im_aspect_ratio"):
        im_aspect_ratio = train_dataset.get_im_aspect_ratio()
    else:
        im_aspect_ratio = [1.0] * len(train_dataset)
    train_sampler = gcv.nn.sampler.SplitSortedBucketSampler(
        im_aspect_ratio,
        batch_size,
        num_parts=hvd.size() if args.horovod else 1,
        part_index=hvd.rank() if args.horovod else 0,
        shuffle=True,
    )
    train_loader = mx.gluon.data.DataLoader(
        train_dataset.transform(
            train_transform(net.short, net.max_size, net, ashape=net.ashape, multi_stage=True)
        ),
        batch_sampler=train_sampler,
        batchify_fn=train_bfn,
        num_workers=args.num_workers,
    )
    val_bfn = Tuple(*[Append() for _ in range(3)])
    short = net.short[-1] if isinstance(net.short, (tuple, list)) else net.short
    # validation use 1 sample per device
    val_loader = mx.gluon.data.DataLoader(
        val_dataset.transform(val_transform(short, net.max_size)),
        num_shards,
        False,
        batchify_fn=val_bfn,
        last_batch="keep",
        num_workers=args.num_workers,
    )
    return train_loader, val_loader


def save_params(net, logger, best_map, current_map, epoch, save_interval, prefix, args):
    current_map = float(current_map)

    if current_map > best_map[0]:
        logger.info(
            "[Epoch {}] mAP {} higher than current best {} saving to {}".format(
                epoch, current_map, best_map, "{:s}_best.params".format(args.save_prefix)
            )
        )
        best_map[0] = current_map

        net.save_parameters(os.path.join(args.sm_save, "{:s}_best.params".format(args.save_prefix)))
        with open(os.path.join(args.sm_save, args.save_prefix) + "_best_map.log", "a") as f:
            f.write("{:04d}:\t{:.4f}\n".format(epoch, current_map))

    if save_interval and (epoch + 1) % save_interval == 0:
        logger.info(
            "[Epoch {}] Saving parameters to {}".format(
                epoch, "{:s}_{:04d}_{:.4f}.params".format(prefix, epoch, current_map)
            )
        )
        net.save_parameters(
            os.path.join(
                args.sm_save,
                "{:s}_{:04d}_{:.4f}.params".format(args.save_prefix, epoch, current_map),
            )
        )


def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    new_batch = []
    for i, data in enumerate(batch):
        if isinstance(data, (list, tuple)):
            new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        else:
            new_data = [data.as_in_context(ctx_list[0])]
        new_batch.append(new_data)
    return new_batch


def validate(net, val_data, ctx, eval_metric, args):
    """Test on validation dataset."""
    clipper = gcv.nn.bbox.BBoxClipToImage()
    eval_metric.reset()
    net.hybridize()

    for batch in val_data:
        batch = split_and_load(batch, ctx_list=ctx)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y, im_scale in zip(*batch):
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(clipper(bboxes, x))
            # rescale to original resolution
            im_scale = im_scale.reshape((-1)).asscalar()
            det_bboxes[-1] *= im_scale
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_bboxes[-1] *= im_scale
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        for det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff in zip(
            det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults
        ):
            eval_metric.update(det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff)
    return eval_metric.get()


def get_lr_at_iter(alpha, lr_warmup_factor=1.0 / 3.0):
    return lr_warmup_factor * (1 - alpha) + alpha


def train(net, train_data, val_data, eval_metric, batch_size, ctx, logger, args):
    """Training pipeline"""
    args.kv_store = "device" if (args.amp and "nccl" in args.kv_store) else args.kv_store
    kv = mx.kvstore.create(args.kv_store)
    net.collect_params().setattr("grad_req", "null")
    net.collect_train_params().setattr("grad_req", "write")
    optimizer_params = {"learning_rate": args.lr, "wd": args.wd, "momentum": args.momentum}
    if args.amp:
        optimizer_params["multi_precision"] = True
    if args.horovod:
        hvd.broadcast_parameters(net.collect_params(), root_rank=0)
        trainer = hvd.DistributedTrainer(
            net.collect_train_params(),  # fix batchnorm, fix first stage, etc...
            "sgd",
            optimizer_params,
        )
    else:
        trainer = gluon.Trainer(
            net.collect_train_params(),  # fix batchnorm, fix first stage, etc...
            "sgd",
            optimizer_params,
            update_on_kvstore=(False if args.amp else None),
            kvstore=kv,
        )

    if args.amp:
        amp.init_trainer(trainer)

    # lr decay policy
    lr_decay = float(args.lr_decay)
    lr_steps = sorted([float(ls) for ls in args.lr_decay_epoch.split(",") if ls.strip()])
    lr_warmup = float(args.lr_warmup)  # avoid int division

    # TODO(zhreshold) losses?
    rpn_cls_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    rpn_box_loss = mx.gluon.loss.HuberLoss(rho=1 / 9.0)  # == smoothl1
    rcnn_cls_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    rcnn_box_loss = mx.gluon.loss.HuberLoss(rho=1.0)  # == smoothl1
    metrics = [
        mx.metric.Loss("RPN_Conf"),
        mx.metric.Loss("RPN_SmoothL1"),
        mx.metric.Loss("RCNN_CrossEntropy"),
        mx.metric.Loss("RCNN_SmoothL1"),
    ]

    rpn_acc_metric = RPNAccMetric()
    rpn_bbox_metric = RPNL1LossMetric()
    rcnn_acc_metric = RCNNAccMetric()
    rcnn_bbox_metric = RCNNL1LossMetric()
    metrics2 = [rpn_acc_metric, rpn_bbox_metric, rcnn_acc_metric, rcnn_bbox_metric]

    logger.info(args)

    if args.verbose:
        logger.info("Trainable parameters:")
        logger.info(net.collect_train_params().keys())
    logger.info("Start training from [Epoch {}]".format(args.start_epoch))
    best_map = [0]
    for epoch in range(args.start_epoch, args.epochs):
        rcnn_task = ForwardBackwardTask(
            net,
            trainer,
            rpn_cls_loss,
            rpn_box_loss,
            rcnn_cls_loss,
            rcnn_box_loss,
            mix_ratio=1.0,
            amp_enabled=args.amp,
        )
        executor = Parallel(args.executor_threads, rcnn_task) if not args.horovod else None
        mix_ratio = 1.0
        net.hybridize()

        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        for metric in metrics:
            metric.reset()
        tic = time.time()
        btic = time.time()
        base_lr = trainer.learning_rate
        rcnn_task.mix_ratio = mix_ratio
        for i, batch in enumerate(train_data):
            if epoch == 0 and i <= lr_warmup:
                # adjust based on real percentage
                new_lr = base_lr * get_lr_at_iter(
                    i / lr_warmup, args.lr_warmup_factor / args.num_gpus
                )
                if new_lr != trainer.learning_rate:
                    if i % args.log_interval == 0:
                        logger.info(
                            "[Epoch 0 Iteration {}] Set learning rate to {}".format(i, new_lr)
                        )
                    trainer.set_learning_rate(new_lr)
            batch = split_and_load(batch, ctx_list=ctx)
            metric_losses = [[] for _ in metrics]
            add_losses = [[] for _ in metrics2]
            if executor is not None:
                for data in zip(*batch):
                    executor.put(data)
            for j in range(len(ctx)):
                if executor is not None:
                    result = executor.get()
                else:
                    result = rcnn_task.forward_backward(list(zip(*batch))[0])
                if (not args.horovod) or hvd.rank() == 0:
                    for k in range(len(metric_losses)):
                        metric_losses[k].append(result[k])
                    for k in range(len(add_losses)):
                        add_losses[k].append(result[len(metric_losses) + k])
            for metric, record in zip(metrics, metric_losses):
                metric.update(0, record)
            for metric, records in zip(metrics2, add_losses):
                for pred in records:
                    metric.update(pred[0], pred[1])
            trainer.step(batch_size)

            # update metrics
            if (
                (not args.horovod or hvd.rank() == 0)
                and args.log_interval
                and not (i + 1) % args.log_interval
            ):
                msg = ",".join(["{}={:.3f}".format(*metric.get()) for metric in metrics + metrics2])
                logger.info(
                    "[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}".format(
                        epoch, i, args.log_interval * args.batch_size / (time.time() - btic), msg
                    )
                )
                btic = time.time()

        if (not args.horovod) or hvd.rank() == 0:
            msg = ",".join(["{}={:.3f}".format(*metric.get()) for metric in metrics])
            logger.info(
                "[Epoch {}] Training cost: {:.3f}, {}".format(epoch, (time.time() - tic), msg)
            )
            if not (epoch + 1) % args.val_interval:
                # consider reduce the frequency of validation to save time
                map_name, mean_ap = validate(net, val_data, ctx, eval_metric, args)
                val_msg = "\n".join(["{}={}".format(k, v) for k, v in zip(map_name, mean_ap)])
                logger.info("[Epoch {}] Validation: \n{}".format(epoch, val_msg))
                current_map = float(mean_ap[-1])
            else:
                current_map = 0.0
            save_params(
                net,
                logger,
                best_map,
                current_map,
                epoch,
                args.save_interval,
                os.path.join(args.sm_save, args.save_prefix),
                args,
            )


if __name__ == "__main__":
    args = parse_args()
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)
    if args.amp:
        amp.init()

    # training contexts
    if args.horovod:
        ctx = [mx.gpu(hvd.local_rank())]
    else:
        ctx = [mx.gpu(int(i)) for i in range(args.gpus)]
        ctx = ctx if ctx else [mx.cpu()]

    # network
    kwargs = {}
    module_list = []
    module_list.append("fpn")

    num_gpus = hvd.size() if args.horovod else len(ctx)
    net_name = "_".join(("faster_rcnn", "fpn", "resnet50_v1b", "coco"))

    train_dataset, val_dataset, eval_metric = get_dataset(args)

    if args.horovod:
        net = get_model(
            net_name,
            root=str(hvd.rank()),
            pretrained_base=True,
            per_device_batch_size=args.batch_size,
            **kwargs
        )
    else:
        net = get_model(
            net_name, pretrained_base=True, per_device_batch_size=args.batch_size, **kwargs
        )
    args.save_prefix += net_name

    if args.resume.strip():
        net.load_parameters(args.resume.strip())
    else:
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize()
    net.collect_params().reset_ctx(ctx)

    if args.amp:
        # Cast both weights and gradients to 'float16'
        net.cast("float16")
        # These layers don't support type 'float16'
        net.collect_params(".*batchnorm.*").setattr("dtype", "float32")
        net.collect_params(".*normalizedperclassboxcenterencoder.*").setattr("dtype", "float32")

    # scale according to gpus
    batch_size = args.batch_size * (num_gpus if not args.horovod else 1)
    args.batch_size = args.batch_size * num_gpus

    args.num_gpus = num_gpus

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + "_train.log"
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    if MPI is None and args.horovod:
        logger.warning("mpi4py is not installed, validation result may be incorrect.")

    logger.info("previous lr {}".format(args.lr))
    args.lr = args.lr * float((num_gpus / hvd.local_size() if args.horovod else 1))
    logger.info("scaled lr {}".format(args.lr))
    # training data

    train_data, val_data = get_dataloader(
        net,
        train_dataset,
        val_dataset,
        FasterRCNNDefaultTrainTransform,
        FasterRCNNDefaultValTransform,
        batch_size,
        len(ctx),
        args,
    )

    # training
    train(net, train_data, val_data, eval_metric, batch_size, ctx, logger, args)
