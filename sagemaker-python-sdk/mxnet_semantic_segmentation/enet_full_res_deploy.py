"""
Copyright [2018]-[2018] Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

    http://aws.amazon.com/apache2.0/

or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

Portions copyright Copyright (c) 2017 Pavlos. Please see LICENSE.txt for applicable license terms and NOTICE.txt for applicable notices.
"""
from __future__ import print_function
import mxnet as mx
from mxnet import ndarray as F
from mxnet.io import DataBatch, DataDesc
import os
import numpy as np
import logging
import urllib
import zipfile
import tarfile
import shutil
import gzip
from glob import glob
import random
import json

###############################
###     Loss Functions      ###
###############################

def dice_coef(y_true, y_pred):
    intersection = mx.sym.sum(mx.sym.broadcast_mul(y_true, y_pred), axis=(1, 2, 3))
    return mx.sym.broadcast_div((2. * intersection + 1.),(mx.sym.sum(y_true, axis=(1, 2, 3)) + mx.sym.sum(y_pred, axis=(1, 2, 3)) + 1.))

def dice_coef_loss(y_true, y_pred):
    intersection = mx.sym.sum(mx.sym.broadcast_mul(y_true, y_pred), axis=1, )
    return -mx.sym.broadcast_div((2. * intersection + 1.),(mx.sym.broadcast_add(mx.sym.sum(y_true, axis=1), mx.sym.sum(y_pred, axis=1)) + 1.))

###############################
###     ENet Architecture   ###
###############################

class SpatialDropout(mx.operator.CustomOp):
    def __init__(self, p, num_filters, ctx):
        self._p = float(p)
        self._num_filters = int(num_filters)
        self._ctx = ctx
        self._spatial_dropout_mask = F.ones(shape=(1, 1, 1, 1), ctx=self._ctx)
        
    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        if is_train:
            self._spatial_dropout_mask = F.broadcast_greater(
                F.random_uniform(low=0, high=1, shape=(1, self._num_filters, 1, 1), ctx=self._ctx), 
                F.ones(shape=(1, self._num_filters, 1, 1), ctx=self._ctx) * self._p,
                ctx=self._ctx
            )
            y = F.broadcast_mul(x, self._spatial_dropout_mask, ctx=self._ctx) / (1-self._p)
            self.assign(out_data[0], req[0], y)
        else:
            self.assign(out_data[0], req[0], x)
            
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        dy = out_grad[0]
        dx = F.broadcast_mul(self._spatial_dropout_mask, dy)
        self.assign(in_grad[0], req[0], dx)
        
@mx.operator.register('spatial_dropout')
class SpatialDropoutProp(mx.operator.CustomOpProp):
    def __init__(self, p, num_filters):
        super(SpatialDropoutProp, self).__init__(True)
        self._p = p
        self._num_filters = num_filters
        
    def infer_shape(self, in_shapes):
        data_shape = in_shapes[0]
        output_shape = data_shape
        # return 3 lists representing inputs shapes, outputs shapes, and aux data shapes.
        return (data_shape,), (output_shape,), ()
            
    def create_operator(self, ctx, in_shape, in_dtypes):
        return SpatialDropout(self._p, self._num_filters, ctx)

#begin third party code
#third party code modified in porting from Keras to MXNet
def same_padding(inp_dims, outp_dims, strides, kernel):
    inp_h, inp_w = inp_dims[1:]
    outp_h, outp_w = outp_dims
    kernel_h, kernel_w = kernel
    pad_along_height = max((outp_h - 1) * strides[0] + kernel_h - inp_h, 0)
    pad_along_width = max((outp_w - 1) * strides[1] + kernel_w - inp_w, 0)
    pad_top = pad_along_height // 2          
    pad_bottom = pad_along_height - pad_top  
    pad_left = pad_along_width // 2          
    pad_right = pad_along_width - pad_left   
    return (0,0,0,0,pad_top,pad_bottom,pad_left, pad_right)

def initial_block(inp, inp_dims, outp_dims, nb_filter=13, nb_row=3, nb_col=3, strides=(2, 2)):
    
    padded_inp = mx.sym.pad(inp, mode='constant',
                            pad_width=same_padding(inp_dims, outp_dims, strides, kernel=(nb_row, nb_col)), name='init_pad')
    conv = mx.sym.Convolution(padded_inp, num_filter=nb_filter, kernel=(nb_row, nb_col), stride=strides, name='init_conv')
    max_pool = mx.sym.Pooling(inp, kernel=(2,2), stride=(2,2), pool_type='max', name='init_pool')
    merged = mx.sym.concat(*[conv, max_pool], dim=1, name='init_concat')
    return merged

def encoder_bottleneck(inp, inp_filter, output, name, internal_scale=4, asymmetric=0, dilated=0, downsample=False, dropout_rate=0.1):
    # main branch
    internal = output // internal_scale
    encoder = inp

    # 1x1
    input_stride = 2 if downsample else 1  # the 1st 1x1 projection is replaced with a 2x2 convolution when downsampling
    encoder = mx.sym.Convolution(encoder, num_filter=internal, kernel=(input_stride, input_stride),
                                stride=(input_stride, input_stride), no_bias=True, name="conv1_%i"%name)
    # Batch normalization + PReLU
    encoder = mx.sym.BatchNorm(encoder, momentum=0.1, name="bn1_%i"%name)
    encoder = mx.sym.LeakyReLU(encoder, act_type='prelu', name='prelu1_%i'%name)
    # conv
    if not asymmetric and not dilated:
        encoder = mx.sym.Convolution(encoder, num_filter=internal, kernel=(3,3), pad=(1,1), name="conv2_%i"%name)
    elif asymmetric:
        encoder = mx.sym.Convolution(encoder, num_filter=internal, kernel=(1, asymmetric),
                                     pad=(0, asymmetric// 2), no_bias=True, name="conv3_%i"%name)
        encoder = mx.sym.Convolution(encoder, num_filter=internal, kernel=(asymmetric, 1),
                                     pad=(asymmetric// 2, 0), name="conv4_%i"%name)
    elif dilated:
        encoder = mx.sym.Convolution(encoder, num_filter=internal, kernel=(3,3),
                                     dilate=(dilated, dilated), pad=((3+(dilated-1)*2)// 2, (3+(dilated-1)*2)// 2),
                                     name="conv2_%i"%name)
    else:
        raise(Exception('You shouldn\'t be here'))
    encoder = mx.sym.BatchNorm(encoder, momentum=0.1, name='bn2_%i'%name)
    encoder = mx.sym.LeakyReLU(encoder, act_type='prelu', name='prelu2_%i'%name)    
    # 1x1
    encoder = mx.sym.Convolution(encoder, num_filter=output, kernel=(1,1), no_bias=True, name="conv5_%i"%name)
    encoder = mx.sym.BatchNorm(encoder, momentum=0.1, name='bn3_%i'%name)
    encoder = mx.sym.Custom(encoder, op_type='spatial_dropout', name='spatial_dropout_%i' % name,
                            p = dropout_rate, num_filters = output)
#     encoder = SpatialDropout(encoder, output, dropout_rate, train, name)
    other = inp
    # other branch
    if downsample:
        other = mx.sym.Pooling(other, kernel=(2,2), stride=(2,2), pool_type='max', name='pool1_%i'%name)
        other = mx.sym.transpose(other, axes=(0,2,1,3), name='trans1_%i'%name)
        pad_feature_maps = output - inp_filter
        other = mx.sym.pad(other, mode='constant', pad_width=(0,0,0,0,0,pad_feature_maps,0,0), name='pad1_%i'%name)
        other = mx.sym.transpose(other, axes=(0,2,1,3), name='trans2_%i'%name)
    encoder = mx.sym.broadcast_add(encoder, other, name='add1_%i'%name)
    encoder = mx.sym.LeakyReLU(encoder, act_type='prelu', name='prelu3_%i'%name)
    return encoder

def build_encoder(inp, inp_dims, dropout_rate=0.01):
    enet = initial_block(inp, inp_dims=inp_dims, outp_dims=(inp_dims[1]//2, inp_dims[2]//2))
    enet = mx.sym.BatchNorm(enet, momentum=0.1, name='bn_0')
    encet = mx.sym.LeakyReLU(enet, act_type='prelu', name='prelu_0')
    enet = encoder_bottleneck(enet, 13+inp_dims[0], 64, downsample=True, dropout_rate=dropout_rate, name=1)  # bottleneck 1.0
    for n in range(4):
        enet = encoder_bottleneck(enet, 64, 64, dropout_rate=dropout_rate, name=n+10)  # bottleneck 1.i
    
    enet = encoder_bottleneck(enet, 64, 128, downsample=True, name=19)  # bottleneck 2.0
    # bottleneck 2.x and 3.x
    for n in range(2):
        enet = encoder_bottleneck(enet, 128, 128, name=n*10+20)  # bottleneck 2.1
        enet = encoder_bottleneck(enet, 128, 128, dilated=2, name=n*10+21)  # bottleneck 2.2
        enet = encoder_bottleneck(enet, 128, 128, asymmetric=5, name=n*10+22)  # bottleneck 2.3
        enet = encoder_bottleneck(enet, 128, 128, dilated=4, name=n*10+23)  # bottleneck 2.4
        enet = encoder_bottleneck(enet, 128, 128, name=n*10+24)  # bottleneck 2.5
        enet = encoder_bottleneck(enet, 128, 128, dilated=8, name=n*10+25)  # bottleneck 2.6
        enet = encoder_bottleneck(enet, 128, 128, asymmetric=5, name=n*10+26)  # bottleneck 2.7
        enet = encoder_bottleneck(enet, 128, 128, dilated=16, name=n*10+27)  # bottleneck 2.8
    return enet

def decoder_bottleneck(encoder, inp_filter, output, upsample=False, upsample_dims=None, reverse_module=False, name=0):
    internal = output // 4
    
    x = mx.sym.Convolution(encoder, num_filter=internal, kernel=(1,1), no_bias=True, name="conv6_%i"%name)
    x = mx.sym.BatchNorm(x, momentum=0.1, name='bn4_%i'%name)
    x = mx.sym.Activation(x, act_type='relu', name='relu1_%i'%name)
    if not upsample:
        x = mx.sym.Convolution(x, num_filter=internal, kernel=(3,3), pad=(1,1), no_bias=False, name="conv7_%i"%name)
    else:
        x = mx.sym.Deconvolution(x, num_filter=internal, kernel=(3, 3), stride=(2, 2), target_shape=upsample_dims, name="dconv1_%i"%name)
    x = mx.sym.BatchNorm(x, momentum=0.1, name='bn5_%i'%name)
    x = mx.sym.Activation(x, act_type='relu', name='relu2_%i'%name)
    x = mx.sym.Convolution(x, num_filter=output, kernel=(1,1), no_bias=True, name="conv8_%i"%name)
    other = encoder
    if inp_filter != output or upsample:
        other = mx.sym.Convolution(other, num_filter=output, kernel=(1,1), no_bias=True, name="conv9_%i"%name)
        other = mx.sym.BatchNorm(other, momentum=0.1, name='bn6_%i'%name)
        if upsample and reverse_module is not False:
            other = mx.sym.UpSampling(other, scale=2, sample_type='nearest', name="upsample1_%i"%name)        
    if upsample and reverse_module is False:
        decoder = x
    else:
        x = mx.sym.BatchNorm(x, momentum=0.1, name='bn7_%i'%name)
        decoder = mx.sym.broadcast_add(x, other, name='add2_%i'%name)
        decoder = mx.sym.Activation(decoder, act_type='relu', name='relu3_%i'%name)
    return decoder

def build_decoder(encoder, nc, output_shape=(3, 512, 512)):
    enet = decoder_bottleneck(encoder, 128, 64, upsample=True, upsample_dims=(output_shape[1]//4, output_shape[2]//4), reverse_module=True, name=20)  # bottleneck 4.0
    enet = decoder_bottleneck(enet, 64, 64, name=21)  # bottleneck 4.1
    enet = decoder_bottleneck(enet, 64, 64, name=22)  # bottleneck 4.2
    enet = decoder_bottleneck(enet, 64, 16, upsample=True, upsample_dims=(output_shape[1]//2, output_shape[2]//2), reverse_module=True, name=23)  # bottleneck 5.0
    enet = decoder_bottleneck(enet, 16, 16, name=24)  # bottleneck 5.1

    enet = mx.sym.Deconvolution(enet, num_filter=nc, kernel=(2, 2), stride=(2, 2), target_shape=(output_shape[1],output_shape[2]), name='dconv2')
    return enet

def build_enet(inp_dims):
    data = mx.sym.Variable(name='data')
    label = mx.sym.Variable(name='label')
    label = mx.sym.flatten(label, name='flat_label')
    encoder = build_encoder(data, inp_dims=inp_dims)
    decoded = build_decoder(encoder, 1, output_shape=inp_dims)
    mask = mx.sym.sigmoid(decoded, name='mask_sigmoid')
    sigmoid = mx.sym.Flatten(mask)
    loss = mx.sym.MakeLoss(dice_coef_loss(label, sigmoid), normalization='batch', name="dice_loss")
    mask_output = mx.sym.BlockGrad(mask, 'mask')
    out = mx.sym.Group([loss, mask_output])
    return out
#end ported third party code

###############################
###     Hosting Methods     ###
###############################

def model_fn(model_dir):
    _, arg_params, aux_params = mx.model.load_checkpoint('%s/model' % model_dir, 0)
    batch_size = 1
    data_shape = (batch_size, 1, 720, 720)
    sym = build_enet(data_shape[1:])
    net = mx.mod.Module(sym, data_names=('data',), label_names=('label',))
    net.bind(data_shapes=[['data', data_shape]], label_shapes=[['label', data_shape]], for_training=False)
    net.set_params(arg_params, aux_params)
    return net