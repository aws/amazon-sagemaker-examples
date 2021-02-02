# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d
from maskrcnn_benchmark.layers.nhwc.misc import Conv2d_NHWC
from maskrcnn_benchmark.layers.nhwc import ConvTranspose2d_NHWC
from maskrcnn_benchmark.layers.nhwc import nhwc_to_nchw_transform, nchw_to_nhwc_transform
from maskrcnn_benchmark.layers.nhwc import init

class MaskRCNNC4Predictor(nn.Module):
    def __init__(self, cfg):
        super(MaskRCNNC4Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]

        if cfg.MODEL.ROI_HEADS.USE_FPN:
            num_inputs = dim_reduced
        else:
            stage_index = 4
            stage2_relative_factor = 2 ** (stage_index - 1)
            res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
            num_inputs = res2_out_channels * stage2_relative_factor
        self.nhwc = cfg.NHWC
        conv = Conv2d_NHWC if self.nhwc else Conv2d
        conv_transpose = ConvTranspose2d_NHWC if self.nhwc else ConvTranspose2d
        self.conv5_mask = conv_transpose(num_inputs, dim_reduced, 2, 2, 0)
        self.mask_fcn_logits = conv(dim_reduced, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                #is_layer_nhwc = self.nhwc and 'conv5' in name
                init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu", nhwc=self.nhwc)

    def forward(self, x):
        #TODO: this transpose may be needed for modularity of Detectron repo
#        if self.nhwc:
#            x = nchw_to_nhwc_transform(x)
        x = F.relu(self.conv5_mask(x))
        logits = self.mask_fcn_logits(x)
        if self.nhwc:
            logits = nhwc_to_nchw_transform(logits)
        return logits


_ROI_MASK_PREDICTOR = {"MaskRCNNC4Predictor": MaskRCNNC4Predictor}


def make_roi_mask_predictor(cfg):
    func = _ROI_MASK_PREDICTOR[cfg.MODEL.ROI_MASK_HEAD.PREDICTOR]
    return func(cfg)
