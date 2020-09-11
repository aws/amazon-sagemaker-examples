'''
This script is to be used solely for the purpose of addressing the labeling job inconsistencies for GT semantic segmentation
tasks between June 4, 2020, and September 4, 2020.
'''

# Installation

'''
pip install -r requirements.txt
'''

# Usage

'''
python sem_seg_fix.py --output-manifest output.manifest
--save-location path/to/local/save/directory --fixed-manifest-location
fixed.manifest


`output.manifest` is an output manifest from the Amazon SageMaker
Ground Truth semantic segmentation labeling job with a single
annotator with flawed annotations (i.e. incorrect label indices in
the pngs). The script will read the entries in this manifest, download
the resulting annotations into memory, fix the annotations using the
label map in the manifest, save those results into
`path/to/local/save/directory`, and also save a modified version of
the output manifest with the location of the saved results in
`fixed.manifest`. The entries in `fixed.manifest` will contain a
`'fixed-label-ref'` field with the absolute (local) location of the
fixed annotations and an `'original-label-ref'` field with the
original s3 URI of the label. The corrected annotations will be saved
in `path/to/local/save/directory/` +
`s3/key/of/original/annotation.png`, that is, the absolute s3 key (without the
bucket name) of the original, flawed annotation.
'''

import argparse
import base64
from collections import defaultdict
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import urlparse

import boto3
import numpy as np
from PIL import Image

s3 = boto3.resource('s3')
try:
    next(iter(s3.buckets.all()))
except Exception as e:
    raise Exception(
        'Could not access your s3 resources. '
        'Please verify that your AWS credentials are correctly configured and '
        'try again.'
    ) from e


class SemSegParser(object):
    def __init__(self, annotation, label_categories):
        self._annotation = annotation
        image_bytes = base64.b64decode(self._annotation)
        img = np.asarray(Image.open(BytesIO(image_bytes)).convert("RGBA"))
        self.hex_colors = defaultdict(str)

        self._img_array, self._class_names, self._label_map = \
            self.get_class_masks(img, label_categories)

    def get_class_masks(self, img, label_categories):
        img_no_alpha = img[:, :, 0:3]
        masks = []
        class_names = []
        rgb_label_maps = self.initialize_label_map()
        for idx_str, category_info in label_categories.items():
            i = int(idx_str)
            class_name = category_info['class-name']
            class_names.append(class_name)
            class_hex_color = category_info['hex-color']
            self.hex_colors[class_name] = class_hex_color
            class_rgb_color = self.hex_to_rgb(class_hex_color)
            rgb_label_maps.append(class_rgb_color)
            class_mask = np.all(img_no_alpha == class_rgb_color, axis=-1)
            class_mask = class_mask * (i+1)
            masks.append(class_mask)
        masks = np.array(masks)
        masks = masks.sum(axis=0)

        return masks.astype(np.uint8), class_names, rgb_label_maps

    # Set background to white
    def initialize_label_map(self):
        return [(255, 255, 255)]

    @property
    def class_names(self):
        return self._class_names

    @property
    def img_array(self):
        return self._img_array

    @staticmethod
    def hex_to_rgb(hexcode):
        h = hexcode.lstrip('#')
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

    @property
    def img_w_palette(self):
        im = Image.fromarray(np.uint8(self._img_array))
        num_classes = len(self._label_map)
        palette = self._label_map + [(255, 255, 255) for i
                                     in range(256 - num_classes)]

        palette = [item for rgb in palette for item in rgb] + ([255, ])
        im.putpalette(palette)
        return im


def get_bucket_and_key(s3uri):
    """Get the bucket name and key associated with an s3 object.

    Args:
      s3uri (str): The s3 uri.

    Return:
      bucket_name and key strings.
    """
    url = urlparse(s3uri)
    bucket_name = url.netloc
    key = url.path.lstrip('/')

    return bucket_name, key


def get_object_bytes(s3_obj):
    """Get bytes for an object stored in s3.

    Arg:
     s3_obj (boto3.resources.factory.s3.ObjectSummary): object for thing in s3
    """
    body = s3_obj.get()['Body']
    return body.read()


def get_metadata(entry):
    return next(iter(
        v for k, v in entry.items() if ('-metadata' in k)
    ))


def fix_annotations(output_manifest, save_location, fixed_manifest_location):
    manifest = []
    with open(output_manifest, 'r') as f:
        for line in f:
            manifest.append(json.loads(line))

    if save_location.endswith('/'):
        save_location = save_location[:-1]

    fixed_manifest = []
    for entry in manifest:
        metadata = get_metadata(entry)
        job_name = metadata['job-name'].replace('labeling-job/', '')
        label_key = job_name + '-ref'
        label_uri = entry[label_key]
        bucket_name, key = get_bucket_and_key(label_uri)
        bucket = s3.Bucket(bucket_name)
        annotation_bytes = get_object_bytes(bucket.Object(key))
        annotation_b64 = base64.b64encode(annotation_bytes)
        color_map = metadata['internal-color-map']
        parser = SemSegParser(annotation_b64, color_map)

        save_dir = Path(save_location)
        save_path = Path.joinpath(save_dir, key)
        try:
            save_path.parent.mkdir(parents=True)
        except FileExistsError as e:
            raise FileExistsError(
                'File already exists at save location for fixed annotation. '
                'Please change save-location and try again.'
            )
        parser.img_w_palette.save(save_path)

        entry['original-label-ref'] = entry[label_key]
        entry['fixed-label-ref'] = str(save_path.absolute())
        fixed_manifest.append(entry)

    Path(fixed_manifest_location).parent.mkdir(exist_ok=True, parents=True)
    with open(fixed_manifest_location, 'w') as f:
        for line in fixed_manifest:
            f.write(json.dumps(line) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Correct semantic segmentation masks from Amazon SageMaker Ground '
            'Truth.'
        )
    )
    parser.add_argument(
        '--output-manifest', type=str, required=True,
        help=(
            'Path to the output manifest from your labeling job.'
        )
    )
    parser.add_argument(
        '--save-location', type=str, default='./output',
        help=(
            'Local directory to save corrected annotations.'
        )
    )
    parser.add_argument(
        '--fixed-manifest-location', type=str, default='fixed.manifest',
        help=(
            'Location to save manifest with fixed annotation information.'
        )
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(
        'Fixing annotations from {} and saving to {}...'.format(
            args.output_manifest, args.save_location,
        )
    )
    fix_annotations(
        args.output_manifest, args.save_location, args.fixed_manifest_location,
    )
    print('Done.')
