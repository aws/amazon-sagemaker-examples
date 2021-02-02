// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
// Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
#pragma once
#include <torch/extension.h>


at::Tensor SigmoidFocalLoss_forward_cuda(
		const at::Tensor& logits,
                const at::Tensor& targets,
		const int num_classes, 
		const float gamma, 
		const float alpha); 

at::Tensor SigmoidFocalLoss_backward_cuda(
			     const at::Tensor& logits,
                             const at::Tensor& targets,
			     const at::Tensor& d_losses,
			     const int num_classes,
			     const float gamma,
			     const float alpha);

at::Tensor ROIAlign_forward_cuda(const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int sampling_ratio,
                                 const bool is_nhwc);

at::Tensor ROIAlign_backward_cuda(const at::Tensor& grad,
                                  const at::Tensor& rois,
                                  const float spatial_scale,
                                  const int pooled_height,
                                  const int pooled_width,
                                  const int batch_size,
                                  const int channels,
                                  const int height,
                                  const int width,
                                  const int sampling_ratio,
                                  const bool is_nhwc);


std::tuple<at::Tensor, at::Tensor> ROIPool_forward_cuda(const at::Tensor& input,
                                const at::Tensor& rois,
                                const float spatial_scale,
                                const int pooled_height,
                                const int pooled_width);

at::Tensor ROIPool_backward_cuda(const at::Tensor& grad,
                                 const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const at::Tensor& argmax,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int batch_size,
                                 const int channels,
                                 const int height,
                                 const int width);

at::Tensor nms_cuda(const at::Tensor boxes, float nms_overlap_thresh);


at::Tensor compute_flow_cuda(const at::Tensor& boxes,
                             const int height,
                             const int width);
                            
at::Tensor generate_mask_targets_cuda(at::Tensor dense_vector, 
                                      const std::vector<std::vector<at::Tensor>> polygons, 
                                      const at::Tensor anchors, 
                                      const int mask_size);                             

at::Tensor box_iou_cuda(at::Tensor box1, at::Tensor box2);
at::Tensor box_iou_cuda_aligned(at::Tensor box1, at::Tensor box2);

std::vector<at::Tensor> box_encode_cuda(at::Tensor boxes, 
                                     at::Tensor anchors, 
                                     float wx, 
                                     float wy, 
                                     float ww, 
                                     float wh);                                       

at::Tensor match_proposals_cuda(at::Tensor match_quality_matrix,
                                bool include_low_quality_matches, 
                                float low_th, 
                                float high_th);       

// initial_pos_mask is the initial positive masks for boxes, 1 if it is kept and 0 otherwise
at::Tensor nms_batched_cuda(const at::Tensor boxes_cat,
                                const std::vector<int> n_boxes_vec, 
                                const at::Tensor n_boxes, const at::Tensor initial_pos_mask, 
                                const float nms_overlap_thresh); 

namespace rpn {
std::vector<at::Tensor> GeneratePreNMSUprightBoxesBatched_cuda(
    const int num_images,
    const int A,
    const int K_max,
    const int max_anchors,
    at::Tensor& hw_array,
    at::Tensor& num_anchors_per_level,
    at::Tensor& sorted_indices, // topK sorted pre_nms_topn indices
    at::Tensor& sorted_scores,  // topK sorted pre_nms_topn scores [N, A, H, W]
    at::Tensor& bbox_deltas,    // [N, A*4, H, W] (full, unsorted / sliced)
    at::Tensor& anchors,        // input (full, unsorted, unsliced)
    at::Tensor& image_shapes,   // (h, w) of images
    const int pre_nms_nboxes,
    const int rpn_min_size,
    const float bbox_xform_clip_default,
    const bool correct_transform_coords);
} //namespace rpn
