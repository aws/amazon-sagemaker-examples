// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
// Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
#include "nms.h"
#include "ROIAlign.h"
#include "ROIPool.h"
#include "SigmoidFocalLoss.h"
#include "generate_mask_targets.h"
#include "box_iou.h"
#include "box_encode.h"
#include "match_proposals.h"
#include "nms_batched.h"
#include "anchor_generator.h"
#include "rpn_generate_proposals_batched.h"
#ifdef WITH_CUDA
#include "cuda/rpn_generate_proposals.h"
//#include "cuda/rpn_generate_proposals_batched.h"
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "non-maximum suppression");
  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
  m.def("roi_pool_forward", &ROIPool_forward, "ROIPool_forward");
  m.def("roi_pool_backward", &ROIPool_backward, "ROIPool_backward");
  m.def("sigmoid_focalloss_forward", &SigmoidFocalLoss_forward, "SigmoidFocalLoss_forward");
  m.def("sigmoid_focalloss_backward", &SigmoidFocalLoss_backward, "SigmoidFocalLoss_backward");
  m.def("generate_mask_targets", &generate_mask_targets, "generate_mask_targets");
  m.def("box_iou", &box_iou, "box_iou");
  m.def("box_iou_aligned", &box_iou_aligned, "box_iou_aligned");
  m.def("box_encode", &box_encode, "box_encode");
  m.def("match_proposals", &match_proposals, "match_proposals");

  m.def("nms_batched", &nms_batched, "nms_batched"); 
  m.def("anchor_generator", &anchor_generator, "anchor_generator");
#ifdef WITH_CUDA
  m.def("GeneratePreNMSUprightBoxesBatched", &GeneratePreNMSUprightBoxesBatched, "RPN Proposal Generation Batched");
#endif
}
