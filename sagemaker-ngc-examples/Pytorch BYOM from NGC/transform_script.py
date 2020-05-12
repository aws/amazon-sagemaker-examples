
import torch
import os
import numpy as np

import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

class ResNet(nn.Module):
    def __init__(self, backbone='resnet50', backbone_path=None):
        super().__init__()
        if backbone == 'resnet18':
            backbone = resnet18(pretrained=not backbone_path)
            self.out_channels = [256, 512, 512, 256, 256, 128]
        elif backbone == 'resnet34':
            backbone = resnet34(pretrained=not backbone_path)
            self.out_channels = [256, 512, 512, 256, 256, 256]
        elif backbone == 'resnet50':
            backbone = resnet50(pretrained=not backbone_path)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        elif backbone == 'resnet101':
            backbone = resnet101(pretrained=not backbone_path)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        else:  # backbone == 'resnet152':
            backbone = resnet152(pretrained=not backbone_path)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        if backbone_path:
            backbone.load_state_dict(torch.load(backbone_path))

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]

        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x

class SSD300(nn.Module):
    def __init__(self, backbone=ResNet('resnet50')):
        super().__init__()

        self.feature_extractor = backbone

        self.label_num = 81  # number of COCO classes
        self._build_additional_features(self.feature_extractor.out_channels)
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        self.loc = []
        self.conf = []

        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            self.loc.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            self.conf.append(nn.Conv2d(oc, nd * self.label_num, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        self._init_weights()

    def _build_additional_features(self, input_size):
        self.additional_blocks = []
        for i, (input_size, output_size, channels) in enumerate(zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])):
            if i < 3:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )

            self.additional_blocks.append(layer)

        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    def _init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1: nn.init.xavier_uniform_(param)

    # Shape the classifier to the view of bboxes
    def bbox_view(self, src, loc, conf):
        ret = []
        for s, l, c in zip(src, loc, conf):
            ret.append((l(s).view(s.size(0), 4, -1), c(s).view(s.size(0), self.label_num, -1)))

        locs, confs = list(zip(*ret))
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs

    def forward(self, x):
        x = self.feature_extractor(x)

        detection_feed = [x]
        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x)

        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)

        # For SSD 300, shall return nbatch x 8732 x {nlabels, nlocs} results, to get bboxes, use decoder
        return locs, confs

# this function tells the endpoint how to make predictions and how to package them to send back
def predict_fn(input_data, model):
    # run prediction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        pred = model(input_data)
    #pred1, pred2 = pred[0].detach().cpu().numpy(),pred[1].detach().cpu().numpy()
    pred_dict = {'pred1':pred[0].detach().cpu().numpy(), 'pred2':pred[1].detach().cpu().numpy()}
    return pred_dict
        
# this function loads our model weights from s3, or if that fails, from the NGC repo
def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SSD300(backbone=ResNet('resnet50'))
    try:
        model_weights = torch.load(os.path.join(model_dir, 'nvidia_ssdpyt_fp32_190826.pt'), map_location='cpu')['model']
        model.to('cpu')
        model.load_state_dict(model_weights)
    except:
        print('using fallback model loading')
        os.system('wget https://api.ngc.nvidia.com/v2/models/nvidia/ssdpyt_fp32/versions/2/files/nvidia_ssdpyt_fp32_190826.pt')
        model_weights = torch.load(os.path.join('nvidia_ssdpyt_fp32_190826.pt'), map_location='cpu')['model']
        model.to('cpu')
        model.load_state_dict(model_weights)
    model.eval()
    return model 

# this function handles our input data and reshapes it back into an image
def input_fn(request_body, request_content_type):
    """this function handles our input data and reshapes it back into an image"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if(request_content_type == 'application/x-npy'):
        try:
            input_data = np.frombuffer(request_body, dtype=np.float64)
        except:
            input_data = np.array(request_body, dtype=np.float64)
    try:
        input_data = torch.tensor(np.reshape(input_data,(1,3,300,300)), dtype=torch.float32, device=device) # this needs to be a torch tensor 
    except:
        input_data = torch.tensor(np.reshape(input_data[16:],(1,3,300,300)), dtype=torch.float32, device=device) # this needs to be a torch tensor 
    return input_data