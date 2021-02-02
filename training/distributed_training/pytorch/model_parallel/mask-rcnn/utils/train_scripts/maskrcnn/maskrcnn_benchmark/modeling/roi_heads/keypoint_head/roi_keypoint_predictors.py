from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark import layers


class KeypointRCNNPredictor(nn.Module):
    def __init__(self, cfg):
        super(KeypointRCNNPredictor, self).__init__()
        input_features = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_LAYERS[-1]
        num_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES
        deconv_kernel = 4
        self.kps_score_lowres = layers.ConvTranspose2d(
            input_features,
            num_keypoints,
            deconv_kernel,
            stride=2,
            padding=deconv_kernel // 2 - 1,
        )
        nn.init.kaiming_normal_(
            self.kps_score_lowres.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.constant_(self.kps_score_lowres.bias, 0)
        self.up_scale = 2

    def forward(self, x):
        x = self.kps_score_lowres(x)
        x = layers.interpolate(
            x, scale_factor=self.up_scale, mode="bilinear", align_corners=False
        )
        return x


_ROI_KEYPOINT_PREDICTOR = {"KeypointRCNNPredictor": KeypointRCNNPredictor}


def make_roi_keypoint_predictor(cfg):
    func = _ROI_KEYPOINT_PREDICTOR[cfg.MODEL.ROI_KEYPOINT_HEAD.PREDICTOR]
    return func(cfg)
