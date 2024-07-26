import json
import os

import argh
import boto3
import numpy as np
from argh import arg
from plotting_funcs import *
from scipy.spatial import distance
from tqdm import tqdm

s3 = boto3.client("s3")


def compute_dist(img_embeds, dist_func=distance.euclidean, obj="Vehicle:1"):
    dists = []
    inds = []
    for i in img_embeds:
        if (i > 0) & (obj in list(img_embeds[i].keys())):
            if obj in list(img_embeds[i - 1].keys()):
                dist = dist_func(
                    img_embeds[i - 1][obj], img_embeds[i][obj]
                )  # distance  between frame at t0 and t1
                dists.append(dist)
                inds.append(i)
    return dists, inds


def get_problem_frames(
    lab_frame,
    flawed_labels,
    size_thresh=0.25,
    iou_thresh=0.4,
    embed=False,
    imgs=None,
    verbose=False,
    embed_std=2,
):
    """
    Function for identifying potentially problematic frames using bounding box size, rolling IoU, and optionally embedding comparison.
    """
    if embed:
        model = torch.hub.load("pytorch/vision:v0.6.0", "resnet18", pretrained=True)
        model.eval()
        modules = list(model.children())[:-1]
        model = nn.Sequential(*modules)

    frame_res = {}
    for obj in list(np.unique(lab_frame.obj)):
        frame_res[obj] = {}
        lframe_len = max(lab_frame["frameid"])
        ann_subframe = lab_frame[lab_frame.obj == obj]
        size_vec = np.zeros(lframe_len + 1)
        size_vec[ann_subframe["frameid"].values] = ann_subframe["height"] * ann_subframe["width"]
        size_diff = np.array(size_vec[:-1]) - np.array(size_vec[1:])
        norm_size_diff = size_diff / np.array(size_vec[:-1])
        norm_size_diff[np.where(np.isnan(norm_size_diff))[0]] = 0
        norm_size_diff[np.where(np.isinf(norm_size_diff))[0]] = 0
        frame_res[obj]["size_diff"] = [int(x) for x in size_diff]
        frame_res[obj]["norm_size_diff"] = [int(x) for x in norm_size_diff]
        try:
            problem_frames = [int(x) for x in np.where(np.abs(norm_size_diff) > size_thresh)[0]]
            if verbose:
                worst_frame = np.argmax(np.abs(norm_size_diff))
                print("Worst frame for", obj, "in", frame, "is: ", worst_frame)
        except:
            problem_frames = []
        frame_res[obj]["size_problem_frames"] = problem_frames

        iou_vec = np.ones(len(np.unique(lab_frame.frameid)))
        for i in lab_frame[lab_frame.obj == obj].frameid[:-1]:
            iou = calc_frame_int_over_union(lab_frame, obj, i)
            iou_vec[i] = iou

        frame_res[obj]["iou"] = iou_vec.tolist()
        inds = [int(x) for x in np.where(iou_vec < iou_thresh)[0]]
        frame_res[obj]["iou_problem_frames"] = inds

        if embed:
            img_crops = {}
            img_embeds = {}

            for j, img in tqdm(enumerate(imgs)):
                img_arr = np.array(img)
                img_embeds[j] = {}
                img_crops[j] = {}
                for i, annot in enumerate(flawed_labels["tracking-annotations"][j]["annotations"]):
                    try:
                        crop = img_arr[
                            annot["top"] : (annot["top"] + annot["height"]),
                            annot["left"] : (annot["left"] + annot["width"]),
                            :,
                        ]
                        new_crop = np.array(Image.fromarray(crop).resize((224, 224)))
                        img_crops[j][annot["object-name"]] = new_crop
                        new_crop = np.reshape(new_crop, (1, 224, 224, 3))
                        new_crop = np.reshape(new_crop, (1, 3, 224, 224))
                        torch_arr = torch.tensor(new_crop, dtype=torch.float)
                        with torch.no_grad():
                            emb = model(torch_arr)
                        img_embeds[j][annot["object-name"]] = emb.squeeze()
                    except:
                        pass

            dists = compute_dist(img_embeds, obj=obj)

            # look for distances that are 2+ standard deviations greater than the mean distance
            prob_frames = np.where(dists > (np.mean(dists) + np.std(dists) * embed_std))[0]
            frame_res[obj]["embed_prob_frames"] = prob_frames.tolist()

    return frame_res


# for frame in tqdm(frame_dict):
@arg("--bucket", help="s3 bucket to retrieve labels from and save result to", default=None)
@arg(
    "--lab_path",
    help="s3 key for labels to be analyzed, an example would look like mot_track_job_results/annotations/consolidated-annotation/output/0/SeqLabel.json",
    default=None,
)
@arg(
    "--size_thresh",
    help="Threshold for identifying allowable percentage size change for a given object between frames",
    default=0.25,
)
@arg(
    "--iou_thresh",
    help="Threshold for identifying the bounding boxes of objects that fall below this IoU metric between frames",
    default=0.4,
)
@arg(
    "--embed",
    help="Perform sequential object bounding box crop embedding comparison. Generates embeddings for the crop of a given object throughout the video and compares them sequentially, requires downloading a model from PyTorch Torchhub",
    default=False,
)
@arg(
    "--imgs",
    help="Path to images to be used for sequential embedding analysis, only required if embed=True",
    default=None,
)
@arg("--save_path", help="s3 key to save quality analysis results to", default=None)
def run_quality_check(
    bucket=None,
    lab_path=None,
    size_thresh=0.25,
    iou_thresh=0.4,
    embed=False,
    imgs=None,
    save_path=None,
):
    """
    Main data quality check utility.
    Designed for use on a single video basis, please provide a SeqLabel.json file to analyze, this can typically be found in
    the s3 output folder for a given Ground Truth Video job under annotations > consolidated-annotation > output
    """

    print("downloading labels")

    s3.download_file(Bucket=bucket, Key=lab_path, Filename="SeqLabel.json")
    #     os.system(f'aws s3 cp s3://{bucket}/{lab_path} SeqLabel.json')

    with open("SeqLabel.json", "r") as f:
        tlabels = json.load(f)
    lab_frame_real = create_annot_frame(tlabels["tracking-annotations"])

    print("Running analysis...")
    frame_res = get_problem_frames(
        lab_frame_real, tlabels, size_thresh=size_thresh, iou_thresh=iou_thresh, embed=embed
    )

    with open("quality_results.json", "w") as f:
        json.dump(frame_res, f)

    print(f"Output saved to s3 path s3://{bucket}/{save_path}")
    s3.upload_file(Bucket=bucket, Key=save_path, Filename="quality_results.json")


#     os.system(f'aws s3 cp quality_results.json s3://{bucket}/{save_path}')


def main():
    parser = argh.ArghParser()
    parser.add_commands([run_quality_check])
    parser.dispatch()


if __name__ == "__main__":
    main()
