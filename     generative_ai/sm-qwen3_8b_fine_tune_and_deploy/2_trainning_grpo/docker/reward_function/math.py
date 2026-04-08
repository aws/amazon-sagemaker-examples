"""
Reward function for GRPO training.
Based on EasyR1 reward function format.
"""

import re
from typing import Any
import difflib

# Metadata
REWARD_NAME = "tagging_reward"
REWARD_TYPE = "batch"


def format_reward(response: str) -> float:
    """Check if the output format is correct."""
    required_categories = [
        "Product Name", "Brand / Line", "Function & Usage", "Ingredients / Materials",
        "Specifications / Model", "Color", "Style & Features", "Use Occasion", "Corresponding Holiday"
    ]
    
    lines = response.strip().split('\n')
    found_categories = 0
    
    for line in lines:
        if '，' in line and '：' in line:
            parts = line.split('，', 1)
            if len(parts) == 2:
                cat_name = parts[1].split('：')[0]
                if cat_name in required_categories:
                    found_categories += 1
    
    return 1.0 if found_categories == 9 else 0.0


def parse_categories(text: str) -> dict:
    """Parse category text into a dictionary."""
    categories = {}
    lines = text.strip().split('\n')
    for line in lines:
        if '，' in line and '：' in line:
            parts = line.split('，', 1)
            if len(parts) == 2:
                cat_name = parts[1].split('：')[0]
                cat_content = parts[1].split('：')[1] if '：' in parts[1] else ''
                tags = [tag.strip() for tag in cat_content.split('；') if tag.strip()]
                categories[cat_name] = tags
    return categories


def fuzzy_match_score(str1: str, str2: str, threshold: float = 0.5) -> float:
    """Calculate fuzzy match score between two strings."""
    if not str1 or not str2:
        return 0.0
    similarity = difflib.SequenceMatcher(None, str1, str2).ratio()
    return similarity if similarity >= threshold else 0.0


def find_best_matches(gt_tags: list, pred_tags: list) -> tuple:
    """Find the best matching prediction tag for each GT tag, return match scores and used prediction indices."""
    matches = []
    used_pred = set()
    
    for gt_tag in gt_tags:
        best_score = 0.0
        best_idx = -1
        
        for i, pred_tag in enumerate(pred_tags):
            if i in used_pred:
                continue
            score = fuzzy_match_score(gt_tag, pred_tag)
            if score > best_score:
                best_score = score
                best_idx = i
        
        if best_score > 0:
            matches.append(best_score)
            used_pred.add(best_idx)
    
    return matches, used_pred


def calculate_metrics(response: str, ground_truth: str) -> dict:
    """Calculate recall, precision, accuracy, match_quality, and formatting metrics."""
    try:
        # Format score
        format_score = format_reward(response)
        
        # Parse GT and prediction results
        gt_categories = parse_categories(ground_truth)
        pred_categories = parse_categories(response)
        
        if not gt_categories or not pred_categories:
            return {
                "recall": 0.0, 
                "precision": 0.0, 
                "accuracy": 0.0, 
                "match_quality": 0.0, 
                "formatting" : format_score
            }
        
        total_recall = 0.0
        total_precision = 0.0
        total_accuracy = 0.0
        total_match_quality = 0.0
        valid_categories = 0
        
        for cat_name in gt_categories:
            gt_tags = gt_categories.get(cat_name, [])
            pred_tags = pred_categories.get(cat_name, [])
            
            if not gt_tags and not pred_tags:
                total_recall += 1.0
                total_precision += 1.0
                total_accuracy += 1.0
                total_match_quality += 1.0
                valid_categories += 1
                continue
            
            if not gt_tags:
                total_precision += 0.0 if pred_tags else 1.0
                total_accuracy += 1.0
                valid_categories += 1
                continue
            
            if not pred_tags:
                total_recall += 0.0
                total_precision += 1.0
                total_accuracy += 0.0
                valid_categories += 1
                continue
            
            # Calculate matches
            matches, used_pred = find_best_matches(gt_tags, pred_tags)
            
            # Recall: proportion of GT tags found
            recall = len(matches) / len(gt_tags) if gt_tags else 0.0
            
            # Precision: penalize extra predicted terms
            extra_preds = len(pred_tags) - len(used_pred)
            precision = 1.0 - (extra_preds / len(pred_tags)) if pred_tags else 0.0
            
            # Accuracy: classification accuracy of matched terms
            accuracy = len(matches) / len(gt_tags) if gt_tags else 0.0
            
            # Match quality: average quality of matches
            match_quality = sum(matches) / len(matches) if matches else 0.0
            
            total_recall += recall
            total_precision += precision
            total_accuracy += accuracy
            total_match_quality += match_quality
            valid_categories += 1
        
        return {
            "recall": total_recall / valid_categories if valid_categories > 0 else 0.0,
            "precision": total_precision / valid_categories if valid_categories > 0 else 0.0,
            "accuracy": total_accuracy / valid_categories if valid_categories > 0 else 0.0,
            "match_quality": total_match_quality / valid_categories if valid_categories > 0 else 0.0,
            "formatting": format_score
        }
        
    except Exception:
        return {
            "recall": 0.0, 
            "precision": 0.0, 
            "accuracy": 0.0, 
            "match_quality": 0.0, 
            "formatting": 0.0
        }


def compute_score(
    reward_inputs: list[dict[str, Any]],
    recall_weight: float = 0.35,
    precision_weight: float = 0.2,
    accuracy_weight: float = 0.35,
    match_quality_weight: float = 0.05,
    formatting_weight: float = 0.05
) -> list[dict[str, float]]:
    """
    Calculate the final reward score.
    
    Args:
        reward_inputs: List of dicts containing response and ground_truth
            [{"response": "...", "ground_truth": "..."}, ...]
        recall_weight: Weight for recall
        precision_weight: Weight for precision (penalizes extra terms)
        accuracy_weight: Weight for accuracy (classification correctness)
        match_quality_weight: Weight for match quality (fuzzy match score)
        formatting_weight: Weight for formatting
    
    Returns:
        List of dicts with individual and overall scores
        [{"overall": 0.8, "recall": 0.9, "precision": 0.7, ...}, ...]
    """
    scores = []
    
    for reward_input in reward_inputs:
        response = reward_input.get("response", "")
        ground_truth = reward_input.get("ground_truth", "")
        
        # Calculate individual metrics
        metrics = calculate_metrics(response, ground_truth)
        
        # Calculate overall weighted score
        overall = (
            recall_weight * metrics["recall"] + 
            precision_weight * metrics["precision"] + 
            accuracy_weight * metrics["accuracy"] + 
            match_quality_weight * metrics["match_quality"] + 
            formatting_weight * metrics["formatting"]
        )
        
        scores.append({
            "overall": overall,
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "accuracy": metrics["accuracy"],
            "match_quality": metrics["match_quality"],
            "formatting": metrics["formatting"]
        })
    
    return scores
