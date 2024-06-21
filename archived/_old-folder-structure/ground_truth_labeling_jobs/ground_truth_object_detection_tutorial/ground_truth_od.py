"""Define classes and functions for interfacing with SageMaker Ground
Truth object detection.

"""

import os

import imageio
import matplotlib.pyplot as plt
import numpy as np


class BoundingBox:
    """Bounding box for an object in an image."""

    def __init__(self, image_id=None, boxdata=None):
        self.image_id = image_id
        if boxdata:
            for datum in boxdata:
                setattr(self, datum, boxdata[datum])

    def __repr__(self):
        return "Box for image {}".format(self.image_id)

    def compute_bb_data(self):
        """Compute the parameters used for IoU."""
        image = self.image
        self.xmin = self.left / image.width
        self.xmax = (self.left + self.width) / image.width
        self.ymin = self.top / image.height
        self.ymax = (self.top + self.height) / image.height


class WorkerBoundingBox(BoundingBox):
    """Bounding box for an object in an image produced by a worker."""

    def __init__(self, image_id=None, worker_id=None, boxdata=None):
        self.worker_id = worker_id
        super().__init__(image_id=image_id, boxdata=boxdata)


class GroundTruthBox(BoundingBox):
    """Bounding box for an object in an image produced by a worker."""

    def __init__(self, image_id=None, oiddata=None, image=None):
        self.image = image
        self.class_name = oiddata[0]
        xmin, xmax, ymin, ymax = [float(datum) for datum in oiddata[1:]]
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        imw = image.width
        imh = image.height
        boxdata = {
            "height": (ymax - ymin) * imh,
            "width": (xmax - xmin) * imw,
            "left": xmin * imw,
            "top": ymin * imh,
        }
        super().__init__(image_id=image_id, boxdata=boxdata)


class BoxedImage:
    """Image with bounding boxes."""

    def __init__(
        self,
        id=None,
        consolidated_boxes=None,
        worker_boxes=None,
        gt_boxes=None,
        uri=None,
        size=None,
    ):
        self.id = id
        self.uri = uri
        if uri:
            self.filename = uri.split("/")[-1]
            self.oid_id = self.filename.split(".")[0]
        else:
            self.filename = None
            self.oid_id = None
        self.local = None
        self.im = None
        if size:
            self.width = size["width"]
            self.depth = size["depth"]
            self.height = size["height"]
            self.shape = self.width, self.height, self.depth
        if consolidated_boxes:
            self.consolidated_boxes = consolidated_boxes
        else:
            self.consolidated_boxes = []
        if worker_boxes:
            self.worker_boxes = worker_boxes
        else:
            self.worker_boxes = []
        if gt_boxes:
            self.gt_boxes = gt_boxes
        else:
            self.gt_boxes = []

    def __repr__(self):
        return "Image{}".format(self.id)

    def n_consolidated_boxes(self):
        """Count the number of consolidated boxes."""
        return len(self.consolidated_boxes)

    def n_worker_boxes(self):
        return len(self.worker_boxes)

    def download(self, directory):
        target_fname = os.path.join(directory, self.uri.split("/")[-1])
        if not os.path.isfile(target_fname):
            os.system(f"aws s3 cp {self.uri} {target_fname}")
        self.local = target_fname

    def imread(self):
        """Cache the image reading process."""
        try:
            return imageio.imread(self.local)
        except OSError:
            print(
                "You need to download this image first. "
                "Use this_image.download(local_directory)."
            )
            raise

    def plot_bbs(self, ax, bbs, img_kwargs, box_kwargs, **kwargs):
        """Master function for plotting images with bounding boxes."""
        img = self.imread()
        ax.imshow(img, **img_kwargs)
        imh, imw, *_ = img.shape
        box_kwargs["fill"] = None
        if kwargs.get("worker", False):
            # Give each worker a color.
            worker_colors = {}
            worker_count = 0
            for bb in bbs:
                worker = bb.worker_id
                if worker not in worker_colors:
                    worker_colors[worker] = "C" + str((9 - worker_count) % 10)
                    worker_count += 1
                rec = plt.Rectangle(
                    (bb.left, bb.top),
                    bb.width,
                    bb.height,
                    edgecolor=worker_colors[worker],
                    **box_kwargs,
                )
                ax.add_patch(rec)
        else:
            for bb in bbs:
                rec = plt.Rectangle((bb.left, bb.top), bb.width, bb.height, **box_kwargs)
                ax.add_patch(rec)
        ax.axis("off")

    def plot_consolidated_bbs(self, ax, img_kwargs={}, box_kwargs={"edgecolor": "blue", "lw": 3}):
        """Plot the consolidated boxes."""
        self.plot_bbs(ax, self.consolidated_boxes, img_kwargs=img_kwargs, box_kwargs=box_kwargs)

    def plot_worker_bbs(self, ax, img_kwargs={}, box_kwargs={"lw": 2}):
        """Plot the individual worker boxes."""
        self.plot_bbs(
            ax, self.worker_boxes, worker=True, img_kwargs=img_kwargs, box_kwargs=box_kwargs
        )

    def plot_gt_bbs(self, ax, img_kwargs={}, box_kwargs={"edgecolor": "lime", "lw": 3}):
        """Plot the ground truth (Open Image Dataset) boxes."""
        self.plot_bbs(ax, self.gt_boxes, img_kwargs=img_kwargs, box_kwargs=box_kwargs)

    def compute_img_confidence(self):
        """Compute the mean bb confidence."""
        if len(self.consolidated_boxes) > 0:
            return np.mean([box.confidence for box in self.consolidated_boxes])
        else:
            return 0

    def compute_iou_bb(self):
        """Compute the mean intersection over union for a collection of
        bounding boxes.
        """

        # Precompute data for the consolidated boxes if necessary.
        for box in self.consolidated_boxes:
            try:
                box.xmin
            except AttributeError:
                box.compute_bb_data()

        # Make the numpy arrays.
        if self.gt_boxes:
            gts = np.vstack([(box.xmin, box.ymin, box.xmax, box.ymax) for box in self.gt_boxes])
        else:
            gts = []
        if self.consolidated_boxes:
            preds = np.vstack(
                [(box.xmin, box.ymin, box.xmax, box.ymax) for box in self.consolidated_boxes]
            )
        else:
            preds = []
        confs = np.array([box.confidence for box in self.consolidated_boxes])

        if len(preds) == 0 and len(gts) == 0:
            return 1.0
        if len(preds) == 0 or len(gts) == 0:
            return 0.0
        preds = preds[np.argsort(confs.flatten())][::-1]

        is_pred_assigned_to_gt = [False] * len(gts)
        pred_areas = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1])
        gt_areas = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
        all_ious = []
        for pred_id, pred in enumerate(preds):
            best_iou = 0
            best_id = -1
            for gt_id, gt in enumerate(gts):
                if is_pred_assigned_to_gt[gt_id]:
                    continue
                x1 = max(gt[0], pred[0])
                y1 = max(gt[1], pred[1])
                x2 = min(gt[2], pred[2])
                y2 = min(gt[3], pred[3])
                iw = max(0, x2 - x1)
                ih = max(0, y2 - y1)
                inter = iw * ih
                iou = inter / (pred_areas[pred_id] + gt_areas[gt_id] - inter)
                if iou > best_iou:
                    best_iou = iou
                    best_id = gt_id
            if best_id != -1:
                is_pred_assigned_to_gt[best_id] = True
                # True positive! Store the IoU.
                all_ious.append(best_iou)
            else:
                # 0 IoU for each unmatched gt (false-negative).
                all_ious.append(0.0)

        # 0 IoU for each unmatched prediction (false-positive).
        all_ious.extend([0.0] * (len(is_pred_assigned_to_gt) - sum(is_pred_assigned_to_gt)))

        return np.mean(all_ious)


def group_miou(imgs):
    """Compute the mIoU for a group of images.

    Args:
      imgs: list of BoxedImages, with consolidated_boxes and gt_boxes.

    Returns:
      mIoU calculated over the bounding boxes in the group.
    """
    # Create a notional BoxedImage with bounding boxes from imgs.
    all_consolidated_boxes = [box for img in imgs for box in img.consolidated_boxes]
    all_gt_boxes = [box for img in imgs for box in img.gt_boxes]
    notional_image = BoxedImage(consolidated_boxes=all_consolidated_boxes, gt_boxes=all_gt_boxes)

    # Compute and return the mIoU.
    return notional_image.compute_iou_bb()
