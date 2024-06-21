"""This script is to be used solely for the purpose of addressing the
labeling job inconsistencies for Ground Truth semantic segmentation
tasks between June 4, 2020, and September 4, 2020.
"""

# Usage

"""
python sagemaker_ground_truth_semantic_segmentation_mask_index_correction.py --labeling-job-name example-job-name

`example-job-name` is an Amazon SageMaker Ground Truth semantic
segmentation labeling job with a single annotator with inconsistent
annotations (i.e. incorrect label indices in the pngs). This script
will locate the output for the labeling job, download the resulting
annotations into memory, fix the annotations using the label map, and
upload the corrected annotations and a new output manifest with
reference to those annotations to S3.

* The corrected annotations will be saved to an S3 location parallel
  to the original annotations:
  <labeling-job-name>/annotations/consolidated-annotation/output-corrected/
* The corrected manifest will be saved to an S3 location parallel to
  the original annotations:
  <labeling-job-name>/manifests/output/output-corrected.manifest
* The entries in `output-corrected.manifest` will contain a
  `'fixed-label-ref'` field with the S3 URI of the fixed annotations
  and an `'original-label-ref'` field with the original s3 URI of the
  label.
"""


import argparse
import base64
import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import boto3
import numpy as np
from PIL import Image

s3 = boto3.resource("s3")
sagemaker = boto3.client("sagemaker")
try:
    next(iter(s3.buckets.all()))
except Exception as e:
    raise Exception(
        "Could not access your s3 resources. "
        "Please verify that your AWS credentials are correctly configured and "
        "try again."
    ) from e


class SemSegParser(object):
    def __init__(self, annotation, label_categories):
        self._annotation = annotation
        image_bytes = base64.b64decode(self._annotation)
        img = np.asarray(Image.open(BytesIO(image_bytes)).convert("RGBA"))
        self.hex_colors = defaultdict(str)
        self.hex_colors["BACKGROUND"] = "#ffffff"

        self._img_array, self._class_names, self._label_map = self.get_class_masks(
            img, label_categories
        )

    def get_class_masks(self, img, label_categories):
        img_no_alpha = img[:, :, 0:3]
        masks = []
        class_names = ["BACKGROUND"]
        rgb_label_maps = self.initialize_label_map()
        for idx_str, category_info in label_categories.items():
            i = int(idx_str)
            class_name = category_info["class-name"]
            if class_name == "BACKGROUND":
                continue
            class_names.append(class_name)
            class_hex_color = category_info["hex-color"]
            self.hex_colors[class_name] = class_hex_color
            class_rgb_color = self.hex_to_rgb(class_hex_color)
            rgb_label_maps.append(class_rgb_color)
            class_mask = np.all(img_no_alpha == class_rgb_color, axis=-1)
            class_mask = class_mask * i
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
        h = hexcode.lstrip("#")
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

    @property
    def img_w_palette(self):
        im = Image.fromarray(np.uint8(self._img_array))
        num_classes = len(self._label_map)
        palette = self._label_map + [(255, 255, 255) for i in range(256 - num_classes)]

        palette = [item for rgb in palette for item in rgb] + (
            [
                255,
            ]
        )
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
    key = url.path.lstrip("/")

    return bucket_name, key


def get_object_bytes(s3_obj):
    """Get bytes for an object stored in s3.

    Arg:
     s3_obj (boto3.resources.factory.s3.ObjectSummary): object for thing in s3
    """
    body = s3_obj.get()["Body"]
    return body.read()


def get_metadata(entry, label_attribute_name):
    metadata_key = "{}-metadata".format(label_attribute_name)
    try:
        metadata = entry[metadata_key]
    except KeyError as e:
        raise KeyError(
            "The metadata_key (derived from the label-attribute-name or "
            "job-name) is missing from this manifest entry. Please specify a "
            "different label-attribute-name."
        ) from e

    return metadata


def get_output_manifest(labeling_job_name):
    description = sagemaker.describe_labeling_job(LabelingJobName=labeling_job_name)
    label_attribute_name = description["LabelAttributeName"]
    manifest_path = description["LabelingJobOutput"]["OutputDatasetS3Uri"]

    bucket_name, key = get_bucket_and_key(manifest_path)
    manifest_bytes = get_object_bytes(s3.Bucket(bucket_name).Object(key))

    manifest = []
    for line in manifest_bytes.decode().splitlines():
        manifest.append(json.loads(line))

    return manifest, label_attribute_name, bucket_name, key


def fix_annotations(labeling_job_name):
    (
        manifest,
        label_attribute_name,
        output_manifest_bucket_name,
        output_manifest_key,
    ) = get_output_manifest(labeling_job_name)

    fixed_manifest = []
    for entry in manifest:
        metadata = get_metadata(entry, label_attribute_name)
        try:
            job_name = metadata["job-name"].replace("labeling-job/", "")
            label_uri = entry[label_attribute_name]

            bucket_name, key = get_bucket_and_key(label_uri)
            key_for_corrected_png = key.replace(
                "consolidated-annotation/output", "consolidated-annotation/output-corrected"
            )
            bucket = s3.Bucket(bucket_name)
            annotation_bytes = get_object_bytes(bucket.Object(key))
            annotation_b64 = base64.b64encode(annotation_bytes)
            color_map = metadata["internal-color-map"]
            parser = SemSegParser(annotation_b64, color_map)

            png_img = parser.img_w_palette
            with BytesIO() as in_mem_file:
                png_img.save(in_mem_file, format="png")
                in_mem_file.seek(0)
                obj = bucket.Object(key_for_corrected_png)
                obj.upload_fileobj(in_mem_file)

            entry["original-label-ref"] = entry[label_attribute_name]
            entry["corrected-label-ref"] = "s3://" + bucket_name + "/" + key_for_corrected_png

        except KeyError:
            continue

        finally:
            fixed_manifest.append(entry)

    # Upload corrected manifest to S3
    with BytesIO() as in_mem_file:
        for line in fixed_manifest:
            line_to_write = json.dumps(line) + "\n"
            in_mem_file.write(bytes(line_to_write.encode("utf-8")))
        in_mem_file.seek(0)

        corrected_manifest_bucket = s3.Bucket(output_manifest_bucket_name)
        corrected_manifest_key = output_manifest_key.replace(
            "output.manifest", "output-corrected.manifest"
        )
        obj = corrected_manifest_bucket.Object(corrected_manifest_key)
        obj.upload_fileobj(in_mem_file)


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Correct semantic segmentation masks from Amazon SageMaker Ground " "Truth.")
    )
    parser.add_argument("--labeling-job-name", type=str, required=True, help=("Your labeling job."))

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Fixing annotations from {}".format(args.labeling_job_name))
    fix_annotations(args.labeling_job_name)
    print("Done.")
