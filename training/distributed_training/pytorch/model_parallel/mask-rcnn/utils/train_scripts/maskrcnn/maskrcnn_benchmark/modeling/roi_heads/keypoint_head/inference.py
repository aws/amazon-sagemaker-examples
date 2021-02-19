import torch
from torch import nn


class KeypointPostProcessor(nn.Module):
    def __init__(self, keypointer=None):
        super(KeypointPostProcessor, self).__init__()
        self.keypointer = keypointer

    def forward(self, x, boxes):
        mask_prob = x

        scores = None
        if self.keypointer:
            mask_prob, scores = self.keypointer(x, boxes)

        assert len(boxes) == 1, "Only non-batched inference supported for now"
        boxes_per_image = [box.bbox.size(0) for box in boxes]
        mask_prob = mask_prob.split(boxes_per_image, dim=0)
        scores = scores.split(boxes_per_image, dim=0)

        results = []
        for prob, box, score in zip(mask_prob, boxes, scores):
            bbox = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            prob = PersonKeypoints(prob, box.size)
            prob.add_field("logits", score)
            bbox.add_field("keypoints", prob)
            results.append(bbox)

        return results


# TODO remove and use only the Keypointer
import numpy as np
import cv2


def heatmaps_to_keypoints(maps, rois):
    """Extract predicted keypoint locations from heatmaps. Output has shape
    (#rois, 4, #keypoints) with the 4 rows corresponding to (x, y, logit, prob)
    for each keypoint.
    """
    # This function converts a discrete image coordinate in a HEATMAP_SIZE x
    # HEATMAP_SIZE image to a continuous keypoint coordinate. We maintain
    # consistency with keypoints_to_heatmap_labels by using the conversion from
    # Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a
    # continuous coordinate.
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]

    widths = rois[:, 2] - rois[:, 0]
    heights = rois[:, 3] - rois[:, 1]
    widths = np.maximum(widths, 1)
    heights = np.maximum(heights, 1)
    widths_ceil = np.ceil(widths)
    heights_ceil = np.ceil(heights)

    # NCHW to NHWC for use with OpenCV
    maps = np.transpose(maps, [0, 2, 3, 1])
    min_size = 0  # cfg.KRCNN.INFERENCE_MIN_SIZE
    num_keypoints = maps.shape[3]
    xy_preds = np.zeros((len(rois), 3, num_keypoints), dtype=np.float32)
    end_scores = np.zeros((len(rois), num_keypoints), dtype=np.float32)
    for i in range(len(rois)):
        if min_size > 0:
            roi_map_width = int(np.maximum(widths_ceil[i], min_size))
            roi_map_height = int(np.maximum(heights_ceil[i], min_size))
        else:
            roi_map_width = widths_ceil[i]
            roi_map_height = heights_ceil[i]
        width_correction = widths[i] / roi_map_width
        height_correction = heights[i] / roi_map_height
        roi_map = cv2.resize(
            maps[i], (roi_map_width, roi_map_height), interpolation=cv2.INTER_CUBIC
        )
        # Bring back to CHW
        roi_map = np.transpose(roi_map, [2, 0, 1])
        # roi_map_probs = scores_to_probs(roi_map.copy())
        w = roi_map.shape[2]
        pos = roi_map.reshape(num_keypoints, -1).argmax(axis=1)
        x_int = pos % w
        y_int = (pos - x_int) // w
        # assert (roi_map_probs[k, y_int, x_int] ==
        #         roi_map_probs[k, :, :].max())
        x = (x_int + 0.5) * width_correction
        y = (y_int + 0.5) * height_correction
        xy_preds[i, 0, :] = x + offset_x[i]
        xy_preds[i, 1, :] = y + offset_y[i]
        xy_preds[i, 2, :] = 1
        end_scores[i, :] = roi_map[np.arange(num_keypoints), y_int, x_int]

    return np.transpose(xy_preds, [0, 2, 1]), end_scores


from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints


class Keypointer(object):
    """
    Projects a set of masks in an image on the locations
    specified by the bounding boxes
    """

    def __init__(self, padding=0):
        self.padding = padding

    def __call__(self, masks, boxes):
        # TODO do this properly
        if isinstance(boxes, BoxList):
            boxes = [boxes]
        assert len(boxes) == 1

        result, scores = heatmaps_to_keypoints(
            masks.detach().cpu().numpy(), boxes[0].bbox.cpu().numpy()
        )
        return torch.from_numpy(result).to(masks.device), torch.as_tensor(scores, device=masks.device)


def make_roi_keypoint_post_processor(cfg):
    keypointer = Keypointer()
    keypoint_post_processor = KeypointPostProcessor(keypointer)
    return keypoint_post_processor
