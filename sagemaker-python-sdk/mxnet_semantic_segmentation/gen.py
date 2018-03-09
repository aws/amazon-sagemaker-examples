"""
Copyright [2018]-[2018] Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

    http://aws.amazon.com/apache2.0/

or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

Portions copyright Copyright (c) 2015 matthewearl. Please see LICENSE.txt for applicable license terms and NOTICE.txt for applicable notices.
"""

import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import cv2
import math

def generate_background(shape):
    num_clusters = np.random.randint(1, 3)
    cluster_values = np.random.choice(np.concatenate([np.arange(75,125),np.arange(200, 255)]),
                                      size=num_clusters, replace=True)
    X, y = make_blobs(300000, centers=num_clusters)
    X = np.clip(((X / 10) * shape[0]).astype(int), 0, shape[0]-1)
    bg = (np.random.rand(shape[0], shape[1])*255 + np.random.normal(size=shape)).astype(np.uint8)
    for i, x in enumerate(X):
        bg[x[0],x[1]] = cluster_values[(y[i]-1)]
    return bg

#Begin third party code
def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = np.matrix([[  c, 0.,  s],
                      [ 0., 1., 0.],
                      [ -s, 0.,  c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = np.matrix([[ 1., 0., 0.],
                      [ 0.,  c, -s],
                      [ 0.,  s,  c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = np.matrix([[  c, -s, 0.],
                      [  s,  c, 0.],
                      [ 0., 0., 1.]]) * M
    return M


def make_affine_transform(from_shape, to_shape, 
                          min_scale, max_scale,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=1.0):
    out_of_bounds = False

    from_size = np.array([[from_shape[1], from_shape[0]]]).T
    to_size = np.array([[to_shape[1], to_shape[0]]]).T

    scale = np.random.uniform((min_scale + max_scale) * 0.5 -
                           (max_scale - min_scale) * 0.5 * scale_variation,
                           (min_scale + max_scale) * 0.5 +
                           (max_scale - min_scale) * 0.5 * scale_variation)
    if scale > max_scale or scale < min_scale:
        out_of_bounds = True
    roll = np.random.uniform(-0.3, 0.3) * rotation_variation
    pitch = np.random.uniform(-0.2, 0.2) * rotation_variation
    yaw = np.random.uniform(-1.2, 1.2) * rotation_variation

    # Compute a bounding box on the skewed input image (`from_shape`).
    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    h, w = from_shape
    corners = np.matrix([[-w, +w, -w, +w],
                            [-h, -h, +h, +h]]) * 0.5
    skewed_size = np.array(np.max(M * corners, axis=1) -
                              np.min(M * corners, axis=1))

    # Set the scale as large as possible such that the skewed and scaled shape
    # is less than or equal to the desired ratio in either dimension.
    scale *= np.min(to_size / skewed_size)

    # Set the translation such that the skewed and scaled image falls within
    # the output shape's bounds.
    trans = (np.random.random((2,1)) - 0.5) * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    if np.any(trans < -0.5) or np.any(trans > 0.5):
        out_of_bounds = True
    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    M *= scale
    M = np.hstack([M, trans + center_to - M * center_from])

    return M, out_of_bounds

#begin modified third party code
def generate_sample(logo, logo_mask, bg_shape=(256, 256)):
    #generate num logos to use
    num_logos = np.random.randint(4, 15)
    logos = [logo.copy() for i in range(num_logos)]
    logo_masks = [logo_mask.copy() for i in range(num_logos)]
    #generate random structured background
    background = generate_background(bg_shape)
    #generate noisy logo
    for i, lg in enumerate(logos):
        lg[lg!=0] = np.round((lg[lg!=0] + np.random.normal(scale=5.0, size=lg.shape)[lg!=0])).astype(np.uint8)
    #generate matrix for random affine transformation
        M, out_of_bounds = make_affine_transform(
                            from_shape=lg.shape,
                            to_shape=background.shape,
                            min_scale=0.1,
                            max_scale=0.5,
                            rotation_variation=2.0,
                            scale_variation=1.0,
                            translation_variation=1.3)
    # apply transformation to both logo and mask
        lg = cv2.warpAffine(lg, M, (background.shape[1], background.shape[0]))
        logo_masks[i] = cv2.warpAffine(logo_masks[i], M, (background.shape[1], background.shape[0]))
    # insert into background image
        background[logo_masks[i] != 0] = lg[logo_masks[i]!=0]
    #merge masks into one
    merged_masks = sum(logo_masks)
    merged_masks = (merged_masks!=0).astype(np.uint8)
    #shift colors in background
    background + np.random.randint(0, 255)
    return background + np.random.randint(0, 255), merged_masks
#end adapted/copied third party code


def generate_dataset(logo, logo_mask, num_samples=1000, shape=(256, 256)):
    X = []
    Y = []
    for i in range(num_samples):
        x, y = generate_sample(logo, logo_mask, bg_shape=shape)
        X.append(x)
        Y.append(y)
    return np.stack(X), np.stack(Y)

