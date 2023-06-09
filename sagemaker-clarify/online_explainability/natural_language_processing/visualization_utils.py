from captum.attr import visualization

import csv
import numpy as np


# This method is a wrapper around the captum that helps produce visualizations for local explanations. It will
# visualize the attributions for the tokens with red or green colors for negative and positive attributions.
def visualization_record(
    attributions,  # list of attributions for the tokens
    text,  # list of tokens
    pred,  # the prediction value obtained from the endpoint
    delta,
    true_label,  # the true label from the dataset
    normalize=True,  # normalizes the attributions so that the max absolute value is 1. Yields stronger colors.
    max_frac_to_show=0.05,  # what fraction of tokens to highlight, set to 1 for all.
    match_to_pred=False,  # whether to limit highlights to red for negative predictions and green for positive ones.
    # By enabling `match_to_pred` you show what tokens contribute to a high/low prediction not those that oppose it.
):
    if normalize:
        attributions = attributions / max(max(attributions), max(-attributions))
    if max_frac_to_show is not None and max_frac_to_show < 1:
        num_show = int(max_frac_to_show * attributions.shape[0])
        sal = attributions
        if pred < 0.5:
            sal = -sal
        if not match_to_pred:
            sal = np.abs(sal)
        top_idxs = np.argsort(-sal)[:num_show]
        mask = np.zeros_like(attributions)
        mask[top_idxs] = 1
        attributions = attributions * mask
    return visualization.VisualizationDataRecord(
        attributions,
        pred,
        int(pred > 0.5),
        true_label,
        attributions.sum() > 0,
        attributions.sum(),
        text,
        delta,
    )


def visualize_result(result, all_labels):
    if not result["explanations"]:
        print(f"No Clarify explanations for the record(s)")
        return
    all_explanations = result["explanations"]["kernel_shap"]
    all_predictions = list(csv.reader(result["predictions"]["data"].splitlines()))

    labels = []
    predictions = []
    explanations = []

    for i, expl in enumerate(all_explanations):
        if expl:
            labels.append(all_labels[i])
            predictions.append(all_predictions[i])
            explanations.append(all_explanations[i])

    attributions_dataset = [
        np.array([attr["attribution"][0] for attr in expl[0]["attributions"]])
        for expl in explanations
    ]
    tokens_dataset = [
        np.array([attr["description"]["partial_text"] for attr in expl[0]["attributions"]])
        for expl in explanations
    ]

    # You can customize the following display settings
    normalize = True
    max_frac_to_show = 1
    match_to_pred = False
    vis = []
    for attr, token, pred, label in zip(attributions_dataset, tokens_dataset, predictions, labels):
        vis.append(
            visualization_record(
                attr, token, float(pred[0]), 0.0, label, normalize, max_frac_to_show, match_to_pred
            )
        )
    _ = visualization.visualize_text(vis)
