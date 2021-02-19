import itertools
from multiprocessing import Pool
import cv2
import numpy as np
from tqdm import tqdm
import torch
import pycocotools.mask as mask_util
from maskrcnn_benchmark.utils.comm import synchronize
from maskrcnn_benchmark.utils.comm import all_gather, get_rank, is_main_process
from maskrcnn_benchmark.data.datasets.evaluation.coco.coco_eval import evaluate_coco

import smdistributed.modelparallel.torch as smp

def expand_masks_np(mask, padding):
    N = mask.shape[0]
    M = mask.shape[-1]
    pad2 = 2 * padding
    scale = float(M + pad2) / M
    padded_mask = np.zeros((N, 1, M + pad2, M + pad2))
    padded_mask[:, :, padding:-padding, padding:-padding] = mask
    return padded_mask, scale

def expand_boxes_np(boxes, scale):
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = np.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp

def paste_mask_in_image_np(mask, box, im_h, im_w, thresh=0.5, padding=1):
    mask = np.expand_dims(mask, 0)
    padded_mask, scale = expand_masks_np(mask[None], padding=padding)
    mask = padded_mask[0, 0]
    box = np.expand_dims(box, 0)
    box = expand_boxes_np(box, scale)[0]
    box = box.astype(np.int32)
    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)
    mask = cv2.resize(mask, (w,h), interpolation=cv2.INTER_LINEAR)
    if thresh >= 0:
        mask = (mask > thresh).astype(np.uint8)
    else:
        mask = (mask * 255).astype(np.uint8)
    im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)

    im_mask[y_0:y_1, x_0:x_1] = mask[
        (y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])
    ]
    return im_mask

class Masker_NP(object):
    def __init__(self, threshold=0.5, padding=1):
        self.threshold = threshold
        self.padding = padding
    
    def __call__(self, masks, boxes, im_w, im_h):
        res = [
            paste_mask_in_image_np(mask, box, im_h, im_w, self.threshold, self.padding)
            for mask, box in zip(masks, boxes)
        ]
        if len(res) > 0:
            res = np.stack(res)
            res = np.expand_dims(res, 1)
        else:
            res = np.zeros((0, 1, masks.shape[-2], masks.shape[-1]))
        return res
    
@smp.step
def test(model, images):
    with torch.no_grad():
        output = model(images)
    return output

def infer_batch(model, images, targets, image_ids, dataset, cfg):
    device = torch.device(cfg.MODEL.DEVICE)
    cpu_device = torch.device("cpu")
    result_dict = {}
    images = images.to(device)
    output = test(model, images)
    with torch.no_grad():
        merged_output = []
        mb = len(output[0])
        mb_outputs = [[] for _ in range(mb)]
        for o in output:
            for i in range(mb):
                mb_outputs[i].append(o[i])
        for out in mb_outputs:
            merged_output.extend(out)
        output = [o.to(cpu_device) for o in merged_output]
    result_dict['masks'] = [prediction.get_field("mask").numpy() for prediction in output]
    result_dict['scores'] = [prediction.get_field("scores").numpy() for prediction in output]
    result_dict['labels'] = [prediction.get_field("labels").numpy() for prediction in output]
    result_dict['mapped_labels'] = [[dataset.contiguous_category_id_to_json_id[i] \
                                     for i in labels] for labels in result_dict['labels']]
    result_dict['img_info'] = [dataset.get_img_info(image_id) for image_id in image_ids]
    output = [prediction.resize((img_info['width'], img_info['height'])) for img_info, prediction in zip(result_dict['img_info'], output)]
    result_dict['boxes'] = [prediction.bbox.numpy() for prediction in output]
    output = [prediction.convert("xywh") for prediction in output]
    result_dict['boxes_wh'] = [prediction.bbox.numpy() for prediction in output]
    return output, result_dict

def process_predictions(masker, masks, boxes, boxes_wh, img_info, mapped_labels, scores):
    image_width = img_info["width"]
    image_height = img_info["height"]
    image_id = img_info["id"]
    masks = masker(masks, boxes, image_width, image_height)
    box_list = boxes_wh.tolist()
    rles = [
            mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
            for mask in masks
        ]
    mask_results = [{"image_id": int(image_id),
                     "category_id": int(mapped_labels[k]),
                     "segmentation": {'size': rle['size'],
                                      'counts': rle['counts'].decode("utf-8")},
                     "score": float(scores[k]),
                     } \
                     for k, rle in enumerate(rles)
                    ]
    bbox_results = [{"image_id": int(image_id),
                    "category_id": int(mapped_labels[k]),
                    "bbox": [float(i) for i in box],
                    "score": float(scores[k]),
                    } \
                    for k, box in enumerate(box_list)]
    
    return mask_results, bbox_results

def process_predictions_mp(args):
    return process_predictions(*args)

def evaluate_coco__mp(args):
    return evaluate_coco(*args)

def infer_coco_eval(model, data_loader, cfg, pool_size=8):
    process_pool = Pool(pool_size)
    eval_iterator = iter(data_loader)
    mapped = []
    masker = Masker_NP()
    for images, targets, image_ids in tqdm(eval_iterator):
        output, res = infer_batch(model, images, targets, image_ids, data_loader.dataset, cfg)
        mask_args = [(masker, masks, boxes, boxes_wh, img_info, mapped_labels, scores) \
                     for masks, boxes, boxes_wh, img_info, mapped_labels, scores in \
                     zip(res['masks'], res['boxes'], res['boxes_wh'], res['img_info'], res['mapped_labels'], res['scores'])]
        mapped.append(process_pool.map_async(process_predictions_mp, mask_args))
    while not all([i.ready() for i in mapped]):
        continue
    coco_results = {}
    mask_results = []
    box_results = []
    eval_res = None
    for i in mapped:
        image_result = i.get()
        for image in image_result:
            mask_results.extend(image[0])
            box_results.extend(image[1])
    process_pool.close()
    synchronize()
    group = smp.DP_GROUP
    coco_results['segm'] = list(itertools.chain.from_iterable(smp.allgather(mask_results, group)))
    coco_results['bbox'] = list(itertools.chain.from_iterable(smp.allgather(box_results, group)))
    if is_main_process():
        eval_pool = Pool(2)
        mask_res = eval_pool.apply_async(evaluate_coco, 
                                         (data_loader.dataset, coco_results, ("bbox",), cfg.OUTPUT_DIR))
        segm_res = eval_pool.apply_async(evaluate_coco, 
                                         (data_loader.dataset, coco_results, ("segm",), cfg.OUTPUT_DIR))
        while not mask_res.ready() and not segm_res.ready():
            continue
        eval_res = {'bbox': mask_res.get().results['bbox']['AP'],
                    'segm': segm_res.get().results['segm']['AP']}
        eval_pool.close()
    synchronize()
    return eval_res