import numpy as np
import json
import boto3
import copy
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from PIL import Image, ImageColor


def query_Type2(image_file_name, endpoint_name, num_predictions=4):

    with open(image_file_name, "rb") as file:
        input_img_rb = file.read()

    client = boto3.client("runtime.sagemaker")
    query_response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/x-image",
        Body=input_img_rb,
        Accept=f'application/json;verbose;n_predictions={num_predictions}'
    )
    # If we remove ';n_predictions={}' from Accept, we get all the predicted boxes.
    query_response = query_response['Body'].read()

    model_predictions = json.loads(query_response)
    normalized_boxes, classes, scores, labels = (
        model_predictions["normalized_boxes"],
        model_predictions["classes"],
        model_predictions["scores"],
        model_predictions["labels"],
    )
    # Substitute the classes index with the classes name
    class_names = [labels[int(idx)] for idx in classes]
    return normalized_boxes, class_names, scores


# Copied from albumentations/augmentations/functional.py
# Follow albumentations.Normalize, which is used in sagemaker_defect_detection/detector.py
def normalize(img, mean, std, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img


def query_Type1(image_file_name, endpoint_name, num_predictions=4):

    with open(image_file_name, "rb") as file:
        input_img_rb = file.read()

    client = boto3.client(service_name="runtime.sagemaker")
    query_response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/x-image",
        Body=input_img_rb
    )
    query_response = query_response["Body"].read()

    model_predictions = json.loads(query_response)['prediction'][:num_predictions]
    class_names = [int(pred[0])+1 for pred in model_predictions]  # +1 for index starts from 1
    scores = [pred[1] for pred in model_predictions]
    normalized_boxes = [pred[2:] for pred in model_predictions]
    return normalized_boxes, class_names, scores


def plot_results(image, bboxes, categories, d):
    # d - dictionary of endpoint responses

    colors = list(ImageColor.colormap.values())
    with Image.open(image) as im:
        image_np = np.array(im)
    fig = plt.figure(figsize=(20, 14))

    n = len(d)

    # Ground truth
    ax1 = fig.add_subplot(2, 3, 1)
    plt.axis('off')
    plt.title('Ground Truth')

    for bbox in bboxes:
        left, bot, right, top = bbox['bbox']
        x, y, w, h = left, bot, right - left, top - bot

        color = colors[hash(bbox['category_id']) % len(colors)]
        rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor=color, facecolor="none")
        ax1.add_patch(rect)
        ax1.text(x, y, "{}".format(categories[bbox['category_id']]),
            bbox=dict(facecolor="white", alpha=0.5))

    ax1.imshow(image_np)

    # Predictions
    counter = 2
    for k, v in d.items():
        axi = fig.add_subplot(2, 3, counter)
        counter += 1

        if "Type2-HPO" in k:
            k = "Type2-HPO"
        elif "Type2" in k:
            k = "Type2"
        elif "Type1-HPO" in k:
            k = "Type1-HPO"
        elif "Type1" in k:
            k = "Type1"
        else:
            print("Un-recognized type")
            exit()

        plt.title(f'Prediction: {k}')
        plt.axis('off')

        for idx in range(len(v['normalized_boxes'])):
            left, bot, right, top = v['normalized_boxes'][idx]
            x, w = [val * image_np.shape[1] for val in [left, right - left]]
            y, h = [val * image_np.shape[0] for val in [bot, top - bot]]
            color = colors[hash(v['classes_names'][idx]) % len(colors)]
            rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor=color, facecolor="none")
            axi.add_patch(rect)
            axi.text(x, y,
                "{} {:.0f}%".format(categories[v['classes_names'][idx]], v['confidences'][idx] * 100),
                bbox=dict(facecolor="white", alpha=0.5),
            )

        axi.imshow(image_np)

    plt.tight_layout()
    plt.savefig("results/"+ image.split('/')[-1])

    plt.show()
