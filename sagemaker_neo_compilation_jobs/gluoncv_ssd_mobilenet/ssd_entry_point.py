import io
import PIL.Image
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ------------------------------------------------------------ #
# Neo host methods                                             #
# ------------------------------------------------------------ #  

def neo_preprocess(payload, content_type):

    def _read_input_shape(signature):
        shape = signature[-1]['shape']
        shape[0] = 1
        return shape

    def _transform_image(image, shape_info):
        # Fetch image size
        input_shape = _read_input_shape(shape_info)

        # Perform color conversion
        if input_shape[-3] == 3:
            # training input expected is 3 channel RGB
            image = image.convert('RGB')
        elif input_shape[-3] == 1:
            # training input expected is grayscale
            image = image.convert('L')
        else:
            # shouldn't get here
            raise RuntimeError('Wrong number of channels in input shape')

        # Resize
        image = np.asarray(image.resize((input_shape[-2], input_shape[-1])))

        # Normalize
        mean_vec = np.array([0.485, 0.456, 0.406])
        stddev_vec = np.array([0.229, 0.224, 0.225])
        image = (image/255- mean_vec)/stddev_vec

        # Transpose
        if len(image.shape) == 2:  # for greyscale image
            image = np.expand_dims(image, axis=2)
        image = np.rollaxis(image, axis=2, start=0)[np.newaxis, :]

        return image

        
    logging.info('Invoking user-defined pre-processing function')

    if content_type != 'image/jpeg':
        raise RuntimeError('Content type must be image/jpeg')
    
    shape_info = [{"shape":[1,3,512,512], "name":"data"}]
    f = io.BytesIO(payload)
    dtest = _transform_image(PIL.Image.open(f), shape_info)
    return {'data':dtest}

    
def neo_postprocess(result):

    logging.info('Invoking user-defined post-processing function')
 
    js = {'prediction':[],'instance':[]}
    for r in result:
        r = np.squeeze(r)
        js['instance'].append(r.tolist())
    idx, score, bbox = js['instance']
    bbox = np.asarray(bbox)
    res = np.hstack((np.column_stack((idx,score)),bbox))
    for r in res:
        js['prediction'].append(r.tolist())
    del js['instance']
    response_body = json.dumps(js)
    content_type = 'application/json'

    return response_body, content_type

# ------------------------------------------------------------ #
# Training methods                                             #
# ------------------------------------------------------------ #  

import glob
import time
import argparse
import warnings
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd


def parse_args():
    parser = argparse.ArgumentParser(description='Train SSD networks.')
    parser.add_argument('--network', type=str, default='ssd_512_mobilenet1.0_voc',
                        help="Network name")
    parser.add_argument('--data-shape', type=int, default=512,
                        help="Input data shape, use 300, 512.")
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training mini-batch size')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers, you can use larger '
                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=int, default=240,
                        help='Training epochs.')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                        'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='160,200',
                        help='epochs at which learning rate decays. default is 160,200.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4')

    return parser.parse_args()


def get_dataloader(net, data_shape, batch_size, num_workers, ctx):
    """Get dataloader."""
    import os

    os.system('pip3 install gluoncv --pre')

    from gluoncv import data as gdata
    from gluoncv.data.batchify import Tuple, Stack, Pad
    from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform

    width, height = data_shape, data_shape
    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width), ctx))
    anchors = anchors.as_in_context(mx.cpu())
    batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
    train_dataset = gdata.RecordFileDetection(os.path.join(os.environ['SM_CHANNEL_TRAIN'], 'train.rec'))
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    return train_loader

def train(net, train_data, ctx, args):
    """Training pipeline"""
    import os

    os.system('pip3 install gluoncv --pre')

    import gluoncv as gcv
    
    net.collect_params().reset_ctx(ctx)
    
    trainer = gluon.Trainer(
            net.collect_params(), 'sgd',
            {'learning_rate': args.lr, 'wd': args.wd, 'momentum': args.momentum}, update_on_kvstore=None)

    # lr decay policy
    lr_decay = float(args.lr_decay)
    lr_steps = sorted([float(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()])

    mbox_loss = gcv.loss.SSDMultiBoxLoss()
    ce_metric = mx.metric.Loss('CrossEntropy')
    smoothl1_metric = mx.metric.Loss('SmoothL1')

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(args)
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]

    for epoch in range(args.start_epoch, args.epochs):
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        ce_metric.reset()
        smoothl1_metric.reset()
        tic = time.time()
        btic = time.time()
        net.hybridize(static_alloc=True, static_shape=True)

        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)

            with autograd.record():
                cls_preds = []
                box_preds = []
                for x in data:
                    cls_pred, box_pred, _ = net(x)
                    cls_preds.append(cls_pred)
                    box_preds.append(box_pred)
                sum_loss, cls_loss, box_loss = mbox_loss(
                    cls_preds, box_preds, cls_targets, box_targets)
                autograd.backward(sum_loss)
            # since we have already normalized the loss, we don't want to normalize
            # by batch-size anymore
            trainer.step(1)
            
            local_batch_size = int(args.batch_size)
            ce_metric.update(0, [l * local_batch_size for l in cls_loss])
            smoothl1_metric.update(0, [l * local_batch_size for l in box_loss])
            if args.log_interval and not (i + 1) % args.log_interval:
                name1, loss1 = ce_metric.get()
                name2, loss2 = smoothl1_metric.get()
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                        epoch, i, args.batch_size/(time.time()-btic), name1, loss1, name2, loss2))
            btic = time.time()

        name1, loss1 = ce_metric.get()
        name2, loss2 = smoothl1_metric.get()
        logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}'.format(
                epoch, (time.time()-tic), name1, loss1, name2, loss2))
        current_map = 0.

    #save model
    net.set_nms(nms_thresh=0.45, nms_topk=400, post_nms=100)
    net(mx.nd.ones((1,3,512,512), ctx=ctx[0]))
    net.export('%s/model' % os.environ['SM_MODEL_DIR'])
    return net

if __name__ == '__main__':
    import os

    os.system('pip3 install gluoncv --pre')

    from gluoncv import model_zoo
    
    args = parse_args()
    
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    net = model_zoo.get_model(args.network, pretrained=False, ctx=ctx)
    net.initialize(ctx=mx.gpu(0))
    train_loader = get_dataloader(net, args.data_shape, args.batch_size, args.num_workers, ctx[0])

    train(net, train_loader, ctx, args)
    
# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #    
    
def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.
    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)
    """
    net = gluon.SymbolBlock.imports(
        '%s/model-symbol.json' % model_dir,
        ['data'],
        '%s/model-0000.params' % model_dir,
    )
   
    return net
    
def transform_fn(net, data, content_type, output_content_type): 
    """
    Transform incoming requests.
    """
    import os

    os.system('pip3 install gluoncv --pre')

    import gluoncv as gcv
    
    #decode json string into numpy array
    data = json.loads(data)
    
    #preprocess image   
    x, image = gcv.data.transforms.presets.ssd.transform_test(mx.nd.array(data), 512)
    
    #check if GPUs area available
    ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()
    
    #load image onto right context
    x = x.as_in_context(ctx)
    
    #perform inference
    class_IDs, scores, bounding_boxes = net(x)
    
    #create list of results
    result = [class_IDs.asnumpy().tolist(), scores.asnumpy().tolist(), bounding_boxes.asnumpy().tolist()]
    
    #decode as json string
    response_body = json.dumps(result)
    return response_body, output_content_type