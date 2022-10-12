import json
import os
import random
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path


def write_line(img_path, width, height, boxes, ids, idx):
    """Create a line for each image with annotations, width, height and image name."""
    # for header, we use minimal length 2, plus width and height
    # with A: 4, B: 5, C: width, D: height
    A = 4
    B = 5
    C = width
    D = height
    # concat id and bboxes
    labels = np.hstack((ids.reshape(-1, 1), boxes)).astype("float")
    # normalized bboxes (recommanded)
    labels[:, (1, 3)] /= float(width)
    labels[:, (2, 4)] /= float(height)
    # flatten
    labels = labels.flatten().tolist()
    str_idx = [str(idx)]
    str_header = [str(x) for x in [A, B, C, D]]
    str_labels = [str(x) for x in labels]
    str_path = [img_path]
    line = "\t".join(str_idx + str_header + str_labels + str_path) + "\n"
    return line


# adapt from __main__ from im2rec.py
def write_lst(output_file, ids, images_annotations):

    all_labels = set()
    image_info = {}
    for entry in images_annotations['images']:
        if entry["id"] in ids:
            image_info[entry["id"]] = entry
    annotations_info = {}  # one annotation for each id (ie., image)
    for entry in images_annotations['annotations']:
        image_id = entry['image_id']
        if image_id in ids:
            if image_id not in annotations_info:
                annotations_info[image_id] = {'boxes': [], 'labels': []}
            annotations_info[image_id]['boxes'].append(entry['bbox'])
            annotations_info[image_id]['labels'].append(entry['category_id'])
            all_labels.add(entry['category_id'])
    labels_list = [label for label in all_labels]
    class_to_idx_mapping = {label: idx for idx, label in enumerate(labels_list)}
    with open(output_file, "w") as fw:
        for i, image_id in enumerate(annotations_info):
            im_info = image_info[image_id]
            image_file = im_info['file_name']
            height = im_info['height']
            width = im_info['width']
            an_info = annotations_info[image_id]
            boxes = np.array(an_info['boxes'])
            labels = np.array([class_to_idx_mapping[label] for label in an_info['labels']])
            line = write_line(image_file, width, height, boxes, labels, i)
            fw.write(line)


def create_lst(data_dir, args, rnd_seed=100):
    """Generate an lst file based on annotations file which is used to convert the input data to .rec format."""
    with open(os.path.join(data_dir, 'annotations.json')) as f:
        images_annotations = json.loads(f.read())

    # Size of each class
    class_ids = defaultdict(list)
    for entry in images_annotations['images']:
        cls_ = entry['file_name'].split('_')[0]
        class_ids[cls_].append(entry['id'])
    print('\ncategory\tnum of images')
    print('---------------')
    for cls_ in class_ids.keys():
        print(f"{cls_}\t{len(class_ids[cls_])}")

    random.seed(rnd_seed)

    # Split train/val/test image ids
    if args.test_ratio:
        test_ids = []
    if args.train_ratio + args.test_ratio < 1.0:
        val_ids = []
    train_ids = []
    for cls_ in class_ids.keys():
        random.shuffle(class_ids[cls_])
        N = len(class_ids[cls_])
        ids = class_ids[cls_]

        sep = int(N * args.train_ratio)
        sep_test = int(N * args.test_ratio)
        if args.train_ratio == 1.0:
            train_ids.extend(ids)
        else:
            if args.test_ratio:
                test_ids.extend(ids[:sep_test])
            if args.train_ratio + args.test_ratio < 1.0:
                val_ids.extend(ids[sep_test + sep:])
            train_ids.extend(ids[sep_test: sep_test + sep])

    write_lst(args.prefix + "_train.lst", train_ids, images_annotations)
    lsts = [args.prefix + "_train.lst"]
    if args.test_ratio:
        write_lst(args.prefix + "_test.lst", test_ids, images_annotations)
        lsts.append(args.prefix + "_test.lst")
    if args.train_ratio + args.test_ratio < 1.0:
        write_lst(args.prefix + "_val.lst", val_ids, images_annotations)
        lsts.append(args.prefix + "_val.lst")

    return lsts


def parse_args():
    """Defines all arguments.

    Returns:
        args object that contains all the params
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create an image list or \
        make a record database by reading from an image list",
    )
    parser.add_argument("prefix", help="prefix of input/output lst and rec files.")
    parser.add_argument("root", help="path to folder containing images.")

    cgroup = parser.add_argument_group("Options for creating image lists")
    cgroup.add_argument(
        "--exts", nargs="+", default=[".jpeg", ".jpg", ".png"], help="list of acceptable image extensions."
    )
    cgroup.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of images to use for training.")
    cgroup.add_argument("--test-ratio", type=float, default=0, help="Ratio of images to use for testing.")
    cgroup.add_argument(
        "--recursive",
        action="store_true",
        help="If true recursively walk through subdirs and assign an unique label\
        to images in each folder. Otherwise only include images in the root folder\
        and give them label 0.",
    )
    args = parser.parse_args()

    args.prefix = os.path.abspath(args.prefix)
    args.root = os.path.abspath(args.root)
    return args


if __name__ == '__main__':

    args = parse_args()
    data_dir = Path(args.root).parent

    lsts = create_lst(data_dir, args)
    print()

    for lst in lsts:
        os.system(f"python3 ./src/im2rec.py {lst} {os.path.join(data_dir, 'images')} --pass-through --pack-label")
        print()
