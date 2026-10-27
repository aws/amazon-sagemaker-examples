# Use this script to convert annotation xmls to a single annotations.json file that will be taken by Jumpstart OD model
# Reference: XML2JSON.py https://linuxtut.com/en/e391e5e6924945b8a852/

import random
import xmltodict
import copy
import json
import glob
import os
from collections import defaultdict


categories = [
    {"id": 1, "name": "crazing"},
    {"id": 2, "name": "inclusion"},
    {"id": 3, "name": "pitted_surface"},
    {"id": 4, "name": "patches"},
    {"id": 5, "name": "rolled-in_scale"},
    {"id": 6, "name": "scratches"},
]


def XML2JSON(xmlFiles, test_ratio=None, rnd_seed=100):
    """ Convert all xmls to annotations.json

    If the test_ratio is not None, convert to two annotations.json files,
    one for train+val, another one for test.
    """

    images = list()
    annotations = list()
    image_id = 1
    annotation_id = 1
    for file in xmlFiles:
        annotation_path = file
        image = dict()
        with open(annotation_path) as fd:
            doc = xmltodict.parse(fd.read(), force_list=('object'))
        filename = str(doc['annotation']['filename'])
        image['file_name'] = filename if filename.endswith('.jpg') else filename + '.jpg'
        image['height'] = int(doc['annotation']['size']['height'])
        image['width'] = int(doc['annotation']['size']['width'])
        image['id'] = image_id
#         print("File Name: {} and image_id {}".format(file, image_id))
        images.append(image)
        if 'object' in doc['annotation']:
            for obj in doc['annotation']['object']:
                for value in categories:
                    annotation = dict()
                    if str(obj['name']) == value["name"]:
                        annotation["image_id"] = image_id
                        xmin = int(obj["bndbox"]["xmin"])
                        ymin = int(obj["bndbox"]["ymin"])
                        xmax = int(obj["bndbox"]["xmax"])
                        ymax = int(obj["bndbox"]["ymax"])
                        annotation["bbox"] = [xmin, ymin, xmax, ymax]
                        annotation["category_id"] = value["id"]
                        annotation["id"] = annotation_id
                        annotation_id += 1
                        annotations.append(annotation)

        else:
            print("File: {} doesn't have any object".format(file))

        image_id += 1

    if test_ratio is None:
        attrDict = dict()
        attrDict["images"] = images
        attrDict["annotations"] = annotations

        jsonString = json.dumps(attrDict)
        with open("annotations.json", "w") as f:
            f.write(jsonString)
    else:
        assert test_ratio < 1.0

        # Size of each class
        category_ids = defaultdict(list)
        for img in images:
            category = img['file_name'].split('_')[0]
            category_ids[category].append(img['id'])
        print('\ncategory\tnum of images')
        print('-' * 20)

        random.seed(rnd_seed)

        train_val_images = []
        test_images = []
        train_val_annotations = []
        test_annotations = []

        for category in category_ids.keys():
            print(f"{category}:\t{len(category_ids[category])}")

            random.shuffle(category_ids[category])
            N = len(category_ids[category])
            ids = category_ids[category]

            sep = int(N * test_ratio)

            category_images = [img for img in images if img['id'] in ids[:sep]]
            test_images.extend(category_images)
            category_images = [img for img in images if img['id'] in ids[sep:]]
            train_val_images.extend(category_images)

            category_annotations = [ann for ann in annotations if ann['image_id'] in ids[:sep]]
            test_annotations.extend(category_annotations)
            category_annotations = [ann for ann in annotations if ann['image_id'] in ids[sep:]]
            train_val_annotations.extend(category_annotations)

        print('-' * 20)

        train_val_attrDict = dict()
        train_val_attrDict["images"] = train_val_images
        train_val_attrDict["annotations"] = train_val_annotations
        print(f"\ntrain_val:\t{len(train_val_images)}")

        train_val_jsonString = json.dumps(train_val_attrDict)
        with open("annotations.json", "w") as f:
            f.write(train_val_jsonString)

        test_attDict = dict()
        test_attDict["images"] = test_images
        test_attDict["annotations"] = test_annotations
        print(f"test:\t{len(test_images)}")

        test_jsonString = json.dumps(test_attDict)
        with open("test_annotations.json", "w") as f:
            f.write(test_jsonString)



def convert_to_pycocotools_ground_truth(annotations_file):
    """
    Given the annotation json file for the test data generated during
    initial data preparatoin, convert it to the input format pycocotools
    can consume.
    """

    with open(annotations_file) as f:
        images_annotations = json.loads(f.read())

    attrDict = dict()
    attrDict["images"] = images_annotations["images"]
    attrDict["categories"] = categories

    annotations = []
    for entry in images_annotations['annotations']:
        ann = copy.deepcopy(entry)
        xmin, ymin, xmax, ymax = ann["bbox"]
        ann["bbox"] = [xmin, ymin, xmax-xmin, ymax-ymin]  # convert to [x, y, W, H]
        ann["area"] = (xmax - xmin) * (ymax - ymin)
        ann["iscrowd"] = 0
        annotations.append(ann)

    attrDict["annotations"] = annotations

    jsonString = json.dumps(attrDict)
    ground_truth_annotations = "results/ground_truth_annotations.json"

    with open(ground_truth_annotations, "w") as f:
        f.write(jsonString)

    return ground_truth_annotations


if __name__ == "__main__":
    data_path = '../NEU-DET/ANNOTATIONS'
    xmlfiles = glob.glob(os.path.join(data_path, '*.xml'))
    xmlfiles.sort()

    XML2JSON(xmlfiles, test_ratio=0.2)


