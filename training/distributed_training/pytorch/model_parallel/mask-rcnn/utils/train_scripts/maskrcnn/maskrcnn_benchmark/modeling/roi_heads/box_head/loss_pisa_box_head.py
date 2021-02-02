# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
import torch

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.layers import GIoULoss
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou, boxlist_iou_batched
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat
from torch.nn.utils.rnn import pad_sequence
from maskrcnn_benchmark.layers import isr_p, carl_loss
from maskrcnn_benchmark.layers import CrossEntropyLoss


class PISALossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
            self,
            proposal_matcher,
            fg_bg_sampler,
            box_coder,
            cls_agnostic_bbox_reg=False,
            decode=False,
            loss="SmoothL1Loss",
            carl=False,
            use_isr_p=False,
            use_isr_n=False
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.giou_loss = GIoULoss(eps=1e-6, reduction="mean", loss_weight=5.0)
        self.cls_loss = CrossEntropyLoss()
        self.decode = decode
        self.loss = loss
        self.carl = carl
        self.use_isr_p = use_isr_p
        self.use_isr_n = use_isr_n

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields("labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def match_targets_to_proposals_batched(self, proposal, target):
        match_quality_matrix = boxlist_iou_batched(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix, batched=1)
        # Fast RCNN only need "labels" field for selecting the targets
        # how to do this for batched case?
        # target = target.copy_with_fields("labels")
        return matched_idxs

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        matched_idxs = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets_per_image = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs_per_image = matched_targets_per_image.get_field("matched_idxs")

            labels_per_image = matched_targets_per_image.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs_per_image == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image.masked_fill_(bg_inds, 0)

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs_per_image == Matcher.BETWEEN_THRESHOLDS
            labels_per_image.masked_fill(ignore_inds, -1)  # -1 is ignored by sampler

            # compute regression targets
            # does not encode target if we need to decode pred later
            if not self.decode:
                regression_targets_per_image = self.box_coder.encode(
                    matched_targets_per_image.bbox, proposals_per_image.bbox
                )
            else:
                regression_targets_per_image = matched_targets_per_image.bbox

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
            matched_idxs.append(matched_idxs_per_image)
        return labels, regression_targets, matched_idxs

    def prepare_targets_batched(self, proposals, targets, target_labels):
        num_images = proposals.size(0)
        matched_idxs = self.match_targets_to_proposals_batched(proposals, targets)
        img_idx = torch.arange(num_images, device=proposals.device)[:, None]
        labels = target_labels[img_idx, matched_idxs.clamp(min=0)]
        labels = labels.to(dtype=torch.int64)
        bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
        labels.masked_fill_(bg_inds, 0)
        ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
        labels.masked_fill_(ignore_inds, -1)

        matched_targets = targets[img_idx, matched_idxs.clamp(min=0)]

        # does not encode target if we need to decode pred later
        if not self.decode:
            regression_targets = self.box_coder.encode(
                matched_targets.view(-1, 4), proposals.view(-1, 4)
            )
        else:
            regression_targets = matched_targets.view(-1, 4)
        return labels, regression_targets.view(num_images, -1, 4), matched_idxs

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """
        num_images = len(proposals[0])
        target_boxes = pad_sequence([target.bbox for target in targets], batch_first=True, padding_value=-1)
        target_labels = pad_sequence([target.get_field("labels") for target in targets], batch_first=True,
                                     padding_value=-1)
        prop_boxes, prop_scores, image_sizes = proposals[0], proposals[1], proposals[2]
        labels, regression_targets, matched_idxs = self.prepare_targets_batched(prop_boxes, target_boxes, target_labels)

        # scores is used as a mask, -1 means box is invalid
        if num_images == 1:
            sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels, is_rpn=0, objectness=prop_scores)
            # when num_images=1, sampled pos inds only has 1 item, so avoid copy in torch.cat
            pos_inds_per_image = [torch.nonzero(sampled_pos_inds[0]).squeeze(1)]
            neg_inds_per_image = [torch.nonzero(sampled_neg_inds[0]).squeeze(1)]
        else:
            sampled_pos_inds, sampled_neg_inds, num_pos_samples, num_neg_samples = self.fg_bg_sampler(labels, is_rpn=0,
                                                                                                      objectness=prop_scores)
            pos_inds_per_image = sampled_pos_inds.split(list(num_pos_samples))
            neg_inds_per_image = sampled_neg_inds.split(list(num_neg_samples))
        prop_boxes = prop_boxes.view(-1, 4)
        regression_targets = regression_targets.view(-1, 4)
        labels = labels.view(-1)
        matched_idxs = matched_idxs.view(-1)
        result_proposals = []
        for i in range(num_images):
            num_pos = pos_inds_per_image[i].size(0)
            num_neg = neg_inds_per_image[i].size(0)
            num_samples = num_pos + num_neg
            inds = torch.cat([pos_inds_per_image[i], neg_inds_per_image[i]])
            box = BoxList(prop_boxes[inds], image_size=image_sizes[i])
            box.add_field("matched_idxs", matched_idxs[inds])
            box.add_field("regression_targets", regression_targets[inds])
            box.add_field("labels", labels[inds])
            label_weights = labels.new_zeros(num_samples)
            target_weights = regression_targets.new_zeros(num_samples, 4)
            if num_pos > 0:
                label_weights[:num_pos] = 1.0
                target_weights[:num_pos, :] = 1.0
            if num_neg > 0:
                label_weights[-num_neg:] = 1.0
            box.add_field("label_weights", label_weights.float())
            box.add_field("target_weights", target_weights.float())
            box.add_field("pos_matched_idxs", matched_idxs[pos_inds_per_image[i]] - 1)
            result_proposals.append(box)
        self._proposals = result_proposals

        return result_proposals

    def __call__(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )

        label_weights = cat(
            [proposal.get_field("label_weights") for proposal in proposals], dim=0
        )

        target_weights = cat(
            [proposal.get_field("target_weights") for proposal in proposals], dim=0
        )

        pos_matched_idxs = [proposal.get_field("pos_matched_idxs") for proposal in proposals]

        rois = torch.cat([a.bbox for a in proposals], dim=0)

        # TODO: get negative sample weights from PISA
        # if neg_label_weights[0] is not None:
        #     label_weights = bbox_targets[1]
        #     cur_num_rois = 0
        #     for i in range(len(sampling_results)):
        #         num_pos = sampling_results[i].pos_inds.size(0)
        #         num_neg = sampling_results[i].neg_inds.size(0)
        #         label_weights[cur_num_rois + num_pos:cur_num_rois + num_pos +
        #                       num_neg] = neg_label_weights[i]
        #         cur_num_rois += num_pos + num_neg

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        pos_label_inds = torch.nonzero(labels > 0).squeeze(1)
        pos_labels = labels.index_select(0, pos_label_inds)
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * pos_labels[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device)

        index_select_indices = ((pos_label_inds[:, None]) * box_regression.size(1) + map_inds).view(-1)
        pos_box_pred_delta = box_regression.view(-1).index_select(0, index_select_indices).view(map_inds.shape[0],
                                                                                                map_inds.shape[1])
        pos_box_target_delta = regression_targets.index_select(0, pos_label_inds)
        pos_rois = rois.index_select(0, pos_label_inds)

        if self.loss == "GIoULoss" and self.decode:
            # target is not encoded
            pos_box_target = pos_box_target_delta
        else:
            pos_box_target = self.box_coder.decode(pos_box_target_delta, pos_rois)
        pos_box_pred = self.box_coder.decode(pos_box_pred_delta, pos_rois)

        # Apply ISR-P
        # use default isr_p config: k=2 bias=0
        bbox_inputs = [labels, label_weights, regression_targets, target_weights, pos_box_pred, pos_box_target,
                       pos_label_inds, pos_labels]

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        if self.use_isr_p:
            labels, label_weights, regression_targets, target_weights = isr_p(
                class_logits,
                bbox_inputs,
                pos_matched_idxs,
                self.cls_loss)
        # end.record()
        # torch.cuda.synchronize()
        # print("isr_p time: ", start.elapsed_time(end))

        avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
        classification_loss = self.cls_loss(class_logits,
                                            labels,
                                            weight=label_weights,
                                            avg_factor=avg_factor
                                            )

        if self.loss == "SmoothL1Loss":
            box_loss = smooth_l1_loss(
                pos_box_pred_delta,
                pos_box_target_delta,
                weight=target_weights,
                size_average=False,
                beta=1,
            )
            box_loss = box_loss / labels.numel()
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            # start.record()
            if self.carl:
                loss_carl = carl_loss(
                    class_logits,
                    pos_label_inds,
                    pos_labels,
                    pos_box_pred_delta,
                    pos_box_target_delta,
                    smooth_l1_loss,
                    k=1,
                    bias=0.2,
                    avg_factor=regression_targets.size(0),
                    num_class=80)
            # end.record()
            # torch.cuda.synchronize()
            # print("carl loss time: ", start.elapsed_time(end))
        elif self.loss == "GIoULoss":
            if pos_box_pred.size()[0] > 0:
                box_loss = self.giou_loss(
                    pos_box_pred,
                    pos_box_target,
                    weight=target_weights.index_select(0, pos_label_inds),
                    avg_factor=labels.numel()
                )
            else:
                box_loss = box_regression.sum() * 0
            if self.carl:
                loss_carl = carl_loss(
                    class_logits,
                    pos_label_inds,
                    pos_labels,
                    pos_box_pred,
                    pos_box_target,
                    self.giou_loss,
                    k=1,
                    bias=0.3,
                    avg_factor=regression_targets.size(0),
                    num_class=80)

        if self.carl:
            return classification_loss, box_loss, loss_carl
        else:
            return classification_loss, box_loss


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    # TODO: add pisa sampler
    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = PISALossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg,
        cfg.MODEL.ROI_BOX_HEAD.DECODE,
        cfg.MODEL.ROI_BOX_HEAD.LOSS,
        cfg.MODEL.ROI_BOX_HEAD.CARL,
        cfg.MODEL.ROI_BOX_HEAD.ISR_P,
        cfg.MODEL.ROI_BOX_HEAD.ISR_N
    )

    return loss_evaluator
