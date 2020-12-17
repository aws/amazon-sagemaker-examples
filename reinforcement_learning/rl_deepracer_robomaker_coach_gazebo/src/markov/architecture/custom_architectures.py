'''This module contains common schemes'''

from markov.architecture.constants import SchemeInfo, Input, ActivationFunctions

# Default network
DEFAULT_INPUT_EMBEDDER = {Input.CAMERA.value: {SchemeInfo.CONV_INFO_LIST.value: [[32, 8, 4],
                                                                                 [64, 4, 2],
                                                                                 [64, 3, 1]],
                                               SchemeInfo.DENSE_LAYER_INFO_LIST.value: [],
                                               SchemeInfo.BN_INFO_CONV.value: [False,
                                                                               ActivationFunctions.RELU.value,
                                                                               0.0],
                                               SchemeInfo.BN_INFO_DENSE.value: [False,
                                                                                ActivationFunctions.RELU.value,
                                                                                0.0],
                                               SchemeInfo.IS_FIRST_LAYER_BN.value: False}}

DEFAULT_MIDDLEWARE = {SchemeInfo.CONV_INFO_LIST.value: [],
                      SchemeInfo.DENSE_LAYER_INFO_LIST.value: [512],
                      SchemeInfo.BN_INFO_CONV.value: [False, ActivationFunctions.RELU.value, 0.0],
                      SchemeInfo.BN_INFO_DENSE.value: [False, ActivationFunctions.RELU.value, 0.0],
                      SchemeInfo.IS_FIRST_LAYER_BN.value: False}

# Shallow network with left and stereo (Left + right)
SHALLOW_LEFT_STEREO_INPUT_EMBEDDER = {
    Input.LEFT_CAMERA.value: {SchemeInfo.CONV_INFO_LIST.value: [[32, 8, 4],
                                                                [64, 4, 2],
                                                                [64, 3, 1]],
                              SchemeInfo.DENSE_LAYER_INFO_LIST.value: [],
                              SchemeInfo.BN_INFO_CONV.value: [False,
                                                              ActivationFunctions.RELU.value,
                                                              0.0],
                              SchemeInfo.BN_INFO_DENSE.value: [False,
                                                               ActivationFunctions.RELU.value,
                                                               0.0],
                              SchemeInfo.IS_FIRST_LAYER_BN.value: False},
    Input.STEREO.value: {SchemeInfo.CONV_INFO_LIST.value: [[32, 8, 4],
                                                           [64, 4, 2],
                                                           [64, 3, 1]],
                         SchemeInfo.DENSE_LAYER_INFO_LIST.value: [],
                         SchemeInfo.BN_INFO_CONV.value: [False,
                                                         ActivationFunctions.RELU.value,
                                                         0.0],
                         SchemeInfo.BN_INFO_DENSE.value: [False,
                                                          ActivationFunctions.RELU.value,
                                                          0.0],
                         SchemeInfo.IS_FIRST_LAYER_BN.value: False}}

# Shallow network with left and stereo (Left + right) and batch norms
SHALLOW_LEFT_STEREO_WITH_BN_INPUT_EMBEDDER = {
    Input.LEFT_CAMERA.value: {SchemeInfo.CONV_INFO_LIST.value: [[32, 8, 4],
                                                                [64, 4, 2],
                                                                [64, 3, 1]],
                              SchemeInfo.DENSE_LAYER_INFO_LIST.value: [],
                              SchemeInfo.BN_INFO_CONV.value: [True,
                                                              ActivationFunctions.RELU.value,
                                                              0.0],
                              SchemeInfo.BN_INFO_DENSE.value: [False,
                                                               ActivationFunctions.RELU.value,
                                                               0.5],
                              SchemeInfo.IS_FIRST_LAYER_BN.value: False},
    Input.STEREO.value: {SchemeInfo.CONV_INFO_LIST.value: [[32, 8, 4],
                                                           [64, 4, 2],
                                                           [64, 3, 1]],
                         SchemeInfo.DENSE_LAYER_INFO_LIST.value: [],
                         SchemeInfo.BN_INFO_CONV.value: [True,
                                                         ActivationFunctions.RELU.value,
                                                         0.0],
                         SchemeInfo.BN_INFO_DENSE.value: [False,
                                                          ActivationFunctions.RELU.value,
                                                          0.5],
                         SchemeInfo.IS_FIRST_LAYER_BN.value: False}}

# Shallow network with only stereo
SHALLOW_STEREO_INPUT_EMBEDDER = {
    Input.STEREO.value: {SchemeInfo.CONV_INFO_LIST.value: [[32, 8, 4],
                                                           [64, 4, 2],
                                                           [64, 3, 1]],
                         SchemeInfo.DENSE_LAYER_INFO_LIST.value: [],
                         SchemeInfo.BN_INFO_CONV.value: [False,
                                                         ActivationFunctions.RELU.value,
                                                         0.0],
                         SchemeInfo.BN_INFO_DENSE.value: [False,
                                                          ActivationFunctions.RELU.value,
                                                          0.0],
                         SchemeInfo.IS_FIRST_LAYER_BN.value: False}
}

# Default lidar network without batch norms
SHALLOW_LIDAR_INPUT_EMBEDDER = {
    Input.CAMERA.value: {SchemeInfo.CONV_INFO_LIST.value: [[32, 8, 4],
                                                           [64, 4, 2],
                                                           [64, 3, 1]],
                         SchemeInfo.DENSE_LAYER_INFO_LIST.value: [],
                         SchemeInfo.BN_INFO_CONV.value: [False,
                                                         ActivationFunctions.RELU.value,
                                                         0.0],
                         SchemeInfo.BN_INFO_DENSE.value: [False,
                                                          ActivationFunctions.RELU.value,
                                                          0.0],
                         SchemeInfo.IS_FIRST_LAYER_BN.value: False},
    Input.LIDAR.value: {SchemeInfo.CONV_INFO_LIST.value: [],
                        SchemeInfo.DENSE_LAYER_INFO_LIST.value: [32, 32],
                        SchemeInfo.BN_INFO_CONV.value: [False,
                                                        ActivationFunctions.RELU.value,
                                                        0.0],
                        SchemeInfo.BN_INFO_DENSE.value: [True,
                                                         ActivationFunctions.RELU.value,
                                                         0.5],
                        SchemeInfo.IS_FIRST_LAYER_BN.value: False}
}

# Default lidar network with stereo left
SHALLOW_LIDAR_SETEREO_LEFT_INPUT_EMBEDDER = {
    Input.LEFT_CAMERA.value: {SchemeInfo.CONV_INFO_LIST.value: [[32, 8, 4],
                                                                [64, 4, 2],
                                                                [64, 3, 1]],
                              SchemeInfo.DENSE_LAYER_INFO_LIST.value: [],
                              SchemeInfo.BN_INFO_CONV.value: [False,
                                                              ActivationFunctions.RELU.value,
                                                              0.3],
                              SchemeInfo.BN_INFO_DENSE.value: [False,
                                                               ActivationFunctions.RELU.value,
                                                               0.0],
                              SchemeInfo.IS_FIRST_LAYER_BN.value: False},
    Input.STEREO.value: {SchemeInfo.CONV_INFO_LIST.value: [[32, 8, 4],
                                                           [64, 4, 2],
                                                           [64, 3, 1]],
                         SchemeInfo.DENSE_LAYER_INFO_LIST.value: [],
                         SchemeInfo.BN_INFO_CONV.value: [False,
                                                         ActivationFunctions.RELU.value,
                                                         0.0],
                         SchemeInfo.BN_INFO_DENSE.value: [False,
                                                          ActivationFunctions.RELU.value,
                                                          0.0],
                         SchemeInfo.IS_FIRST_LAYER_BN.value: False},
    Input.LIDAR.value: {SchemeInfo.CONV_INFO_LIST.value: [],
                        SchemeInfo.DENSE_LAYER_INFO_LIST.value: [32, 32],
                        SchemeInfo.BN_INFO_CONV.value: [False,
                                                        ActivationFunctions.RELU.value,
                                                        0.0],
                        SchemeInfo.BN_INFO_DENSE.value: [False,
                                                         ActivationFunctions.RELU.value,
                                                         0.5],
                        SchemeInfo.IS_FIRST_LAYER_BN.value: False}
}

# Default lidar network, use BN_SCHEME when creating the embedder, embedder activation should be
# NONE
SHALLOW_LIDAR_BN_INPUT_EMBEDDER = {
    Input.CAMERA.value: {SchemeInfo.CONV_INFO_LIST.value: [[32, 8, 4],
                                                           [32, 4, 2],
                                                           [64, 4, 2],
                                                           [64, 3, 1]],
                         SchemeInfo.DENSE_LAYER_INFO_LIST.value: [512, 512],
                         SchemeInfo.BN_INFO_CONV.value: [True,
                                                         ActivationFunctions.RELU.value,
                                                         0.0],
                         SchemeInfo.BN_INFO_DENSE.value: [False,
                                                          ActivationFunctions.RELU.value,
                                                          0.5],
                         SchemeInfo.IS_FIRST_LAYER_BN.value: False},
    Input.LIDAR.value: {SchemeInfo.CONV_INFO_LIST.value: [],
                        SchemeInfo.DENSE_LAYER_INFO_LIST.value: [256, 256],
                        SchemeInfo.BN_INFO_CONV.value: [False,
                                                        ActivationFunctions.RELU.value,
                                                        0.0],
                        SchemeInfo.BN_INFO_DENSE.value: [False,
                                                         ActivationFunctions.RELU.value,
                                                         0.5],
                        SchemeInfo.IS_FIRST_LAYER_BN.value: False}
}

SHALLOW_LIDAR_MIDDLEWARE = {SchemeInfo.CONV_INFO_LIST.value: [],
                            SchemeInfo.DENSE_LAYER_INFO_LIST.value: [256, 128],
                            SchemeInfo.BN_INFO_CONV.value: [False, ActivationFunctions.RELU.value, 0.0],
                            SchemeInfo.BN_INFO_DENSE.value: [False, ActivationFunctions.RELU.value, 0.5],
                            SchemeInfo.IS_FIRST_LAYER_BN.value: False}

# Deep Scheme, use default middle ware, use BN_SCHEME when creating the embedder.
# embedder activation should be RELU
DEEP_INPUT_EMBEDDER = {Input.CAMERA.value: {SchemeInfo.CONV_INFO_LIST.value: [[32, 5, 2],
                                                                                   [32, 3, 1],
                                                                                   [64, 3, 2],
                                                                                   [64, 3, 1]],
                                            SchemeInfo.DENSE_LAYER_INFO_LIST.value: [64],
                                            SchemeInfo.BN_INFO_CONV.value: [True,
                                                                            ActivationFunctions.TANH.value,
                                                                            0.0],
                                            SchemeInfo.BN_INFO_DENSE.value: [False,
                                                                             ActivationFunctions.TANH.value,
                                                                             0.3],
                                            SchemeInfo.IS_FIRST_LAYER_BN.value: False}}

DEEP_LEFT_STEREO_INPUT_EMBEDDER = {
    Input.LEFT_CAMERA.value: {SchemeInfo.CONV_INFO_LIST.value: [[32, 8, 4],
                                                                [64, 4, 2],
                                                                [64, 3, 1]],
                              SchemeInfo.DENSE_LAYER_INFO_LIST.value: [],
                              SchemeInfo.BN_INFO_CONV.value: [False,
                                                              ActivationFunctions.RELU.value,
                                                              0.0],
                              SchemeInfo.BN_INFO_DENSE.value: [False,
                                                               ActivationFunctions.RELU.value,
                                                               0.0],
                              SchemeInfo.IS_FIRST_LAYER_BN.value: False},
    Input.STEREO.value: {SchemeInfo.CONV_INFO_LIST.value: [[32, 3, 1],
                                                           [64, 3, 2],
                                                           [64, 3, 1],
                                                           [128, 3, 2],
                                                           [128, 3, 1]],
                        SchemeInfo.DENSE_LAYER_INFO_LIST.value: [],
                        SchemeInfo.BN_INFO_CONV.value: [False,
                                                        ActivationFunctions.RELU.value,
                                                        0.0],
                        SchemeInfo.BN_INFO_DENSE.value: [False,
                                                         ActivationFunctions.RELU.value,
                                                         0.0],
                        SchemeInfo.IS_FIRST_LAYER_BN.value: False}}

DEEP_STEREO_INPUT_EMBEDDER = {
    Input.STEREO.value: {SchemeInfo.CONV_INFO_LIST.value: [[32, 3, 1],
                                                           [64, 3, 2],
                                                           [64, 3, 1],
                                                           [128, 3, 2],
                                                           [128, 3, 1]],
                        SchemeInfo.DENSE_LAYER_INFO_LIST.value: [],
                        SchemeInfo.BN_INFO_CONV.value: [False,
                                                        ActivationFunctions.RELU.value,
                                                        0.0],
                        SchemeInfo.BN_INFO_DENSE.value: [False,
                                                         ActivationFunctions.RELU.value,
                                                         0.0],
                        SchemeInfo.IS_FIRST_LAYER_BN.value: False}}

#VGG like Scheme, use BN_SCHEME when creating the embedder, embedder activation should be
# NONE
VGG_INPUT_EMBEDDER = {Input.CAMERA.value: {SchemeInfo.CONV_INFO_LIST.value: [[32, 8, 4],
                                                                                  [32, 4, 2],
                                                                                  [64, 4, 2],
                                                                                  [64, 3, 1]],
                                                SchemeInfo.DENSE_LAYER_INFO_LIST.value: [512, 512],
                                                SchemeInfo.BN_INFO_CONV.value: [True,
                                                                                ActivationFunctions.RELU.value,
                                                                                0.0],
                                                SchemeInfo.BN_INFO_DENSE.value: [False,
                                                                                 ActivationFunctions.RELU.value,
                                                                                 0.5],
                                                SchemeInfo.IS_FIRST_LAYER_BN.value: False}}


VGG_LEFT_STEREO_INPUT_EMBEDDER = {Input.CAMERA.value: {SchemeInfo.CONV_INFO_LIST.value: [[32, 8, 4],
                                                                                  [32, 4, 2],
                                                                                  [64, 4, 2],
                                                                                  [64, 3, 1]],
                                                SchemeInfo.DENSE_LAYER_INFO_LIST.value: [512, 512],
                                                SchemeInfo.BN_INFO_CONV.value: [True,
                                                                                ActivationFunctions.RELU.value,
                                                                                0.0],
                                                SchemeInfo.BN_INFO_DENSE.value: [False,
                                                                                 ActivationFunctions.RELU.value,
                                                                                 0.5],
                                                SchemeInfo.IS_FIRST_LAYER_BN.value: False},
                                  Input.STEREO.value: {SchemeInfo.CONV_INFO_LIST.value: [[32, 3, 1],
                                                                                             [64, 3, 2],
                                                                                             [64, 3, 1],
                                                                                             [128, 3, 2],
                                                                                             [128, 3, 1]],
                                                          SchemeInfo.DENSE_LAYER_INFO_LIST.value: [],
                                                          SchemeInfo.BN_INFO_CONV.value: [False,
                                                                                          ActivationFunctions.RELU.value,
                                                                                          0.0],
                                                          SchemeInfo.BN_INFO_DENSE.value: [False,
                                                                                           ActivationFunctions.RELU.value,
                                                                                           0.0],
                                                          SchemeInfo.IS_FIRST_LAYER_BN.value: False}}


VGG_STEREO_INPUT_EMBEDDER = {Input.STEREO.value: {SchemeInfo.CONV_INFO_LIST.value: [[32, 3, 1],
                                                                                   [64, 3, 2],
                                                                                   [64, 3, 1],
                                                                                   [128, 3, 2],
                                                                                   [128, 3, 1]],
                                                SchemeInfo.DENSE_LAYER_INFO_LIST.value: [],
                                                SchemeInfo.BN_INFO_CONV.value: [False,
                                                                                ActivationFunctions.RELU.value,
                                                                                0.0],
                                                SchemeInfo.BN_INFO_DENSE.value: [False,
                                                                                 ActivationFunctions.RELU.value,
                                                                                 0.0],
                                                SchemeInfo.IS_FIRST_LAYER_BN.value: False}}                                               


VGG_MIDDLEWARE = {SchemeInfo.CONV_INFO_LIST.value: [],
                  SchemeInfo.DENSE_LAYER_INFO_LIST.value: [256, 128],
                  SchemeInfo.BN_INFO_CONV.value: [False, ActivationFunctions.RELU.value, 0.0],
                  SchemeInfo.BN_INFO_DENSE.value: [False, ActivationFunctions.RELU.value, 0.5],
                  SchemeInfo.IS_FIRST_LAYER_BN.value: False}

# This archetecture was labled default 7
DEF7_INPUT_EMBEDDER = {Input.CAMERA.value: {SchemeInfo.CONV_INFO_LIST.value: [[32, 3, 2],
                                                                                   [32, 3, 1],
                                                                                   [64, 3, 2],
                                                                                   [64, 3, 1],
                                                                                   [128, 3, 2],
                                                                                   [128, 3, 1],
                                                                                   [256, 3, 2]],
                                                 SchemeInfo.DENSE_LAYER_INFO_LIST.value: [512, 512],
                                                 SchemeInfo.BN_INFO_CONV.value: [True,
                                                                                 ActivationFunctions.RELU.value,
                                                                                 0.0],
                                                 SchemeInfo.BN_INFO_DENSE.value: [False,
                                                                                  ActivationFunctions.RELU.value,
                                                                                  0.5],
                                                 SchemeInfo.IS_FIRST_LAYER_BN.value: False}}

DEF7_MIDDLEWARE = {SchemeInfo.CONV_INFO_LIST.value: [],
                   SchemeInfo.DENSE_LAYER_INFO_LIST.value: [256, 128],
                   SchemeInfo.BN_INFO_CONV.value: [False, ActivationFunctions.RELU.value, 0.0],
                   SchemeInfo.BN_INFO_DENSE.value: [False, ActivationFunctions.RELU.value, 0.5],
                   SchemeInfo.IS_FIRST_LAYER_BN.value: False}

# This archetecture was labled default 5
DEF5_INPUT_EMBEDDER = {Input.CAMERA.value: {SchemeInfo.CONV_INFO_LIST.value: [[32, 3, 2],
                                                                                   [32, 3, 1],
                                                                                   [64, 3, 2],
                                                                                   [64, 3, 1],
                                                                                   [128, 3, 2]],
                                                 SchemeInfo.DENSE_LAYER_INFO_LIST.value: [512, 512],
                                                 SchemeInfo.BN_INFO_CONV.value: [True,
                                                                                 ActivationFunctions.RELU.value,
                                                                                 0.0],
                                                 SchemeInfo.BN_INFO_DENSE.value: [False,
                                                                                  ActivationFunctions.RELU.value,
                                                                                  0.5],
                                                 SchemeInfo.IS_FIRST_LAYER_BN.value: False}}

DEF5_MIDDLEWARE = {SchemeInfo.CONV_INFO_LIST.value: [],
                   SchemeInfo.DENSE_LAYER_INFO_LIST.value: [256, 128],
                   SchemeInfo.BN_INFO_CONV.value: [False, ActivationFunctions.RELU.value, 0.0],
                   SchemeInfo.BN_INFO_DENSE.value: [False, ActivationFunctions.RELU.value, 0.5],
                   SchemeInfo.IS_FIRST_LAYER_BN.value: False}
