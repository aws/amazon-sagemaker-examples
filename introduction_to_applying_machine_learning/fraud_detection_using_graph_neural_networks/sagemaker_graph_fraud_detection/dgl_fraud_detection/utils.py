import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import networkx as nx
import matplotlib.pyplot as plt


def get_metrics(pred, pred_proba, labels, mask, out_dir):
    labels, mask = labels.asnumpy().flatten().astype(int), mask.asnumpy().flatten().astype(int)
    labels, pred, pred_proba = labels[np.where(mask)], pred[np.where(mask)], pred_proba[np.where(mask)]

    acc = ((pred == labels)).sum() / mask.sum()

    true_pos = (np.where(pred == 1, 1, 0) + np.where(labels == 1, 1, 0) > 1).sum()
    false_pos = (np.where(pred == 1, 1, 0) + np.where(labels == 0, 1, 0) > 1).sum()
    false_neg = (np.where(pred == 0, 1, 0) + np.where(labels == 1, 1, 0) > 1).sum()
    true_neg = (np.where(pred == 0, 1, 0) + np.where(labels == 0, 1, 0) > 1).sum()

    precision = true_pos/(true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos/(true_pos + false_neg) if (true_pos + false_neg) > 0 else 0

    f1 = 2*(precision*recall)/(precision + recall) if (precision + recall) > 0 else 0
    confusion_matrix = pd.DataFrame(np.array([[true_pos, false_pos], [false_neg, true_neg]]),
                                    columns=["labels positive", "labels negative"],
                                    index=["predicted positive", "predicted negative"])

    ap = average_precision_score(labels, pred_proba)

    fpr, tpr, _ = roc_curve(labels, pred_proba)
    prc, rec, _ = precision_recall_curve(labels, pred_proba)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(rec, prc)

    save_roc_curve(fpr, tpr, roc_auc, os.path.join(out_dir, "roc_curve.png"))
    save_pr_curve(prc, rec, pr_auc, ap, os.path.join(out_dir, "pr_curve.png"))

    return acc, f1, precision, recall, roc_auc, pr_auc, ap, confusion_matrix


def save_roc_curve(fpr, tpr, roc_auc, location):
    f = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Model ROC curve')
    plt.legend(loc="lower right")
    f.savefig(location)


def save_pr_curve(fpr, tpr, pr_auc, ap, location):
    f = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Model PR curve: AP={0:0.2f}'.format(ap))
    plt.legend(loc="lower right")
    f.savefig(location)


def save_graph_drawing(g, location):
    plt.figure(figsize=(12, 8))
    node_colors = {node: 0.0 if 'user' in node else 0.5 for node in g.nodes()}
    nx.draw(g, node_size=10000, pos=nx.spring_layout(g), with_labels=True, font_size=14,
            node_color=list(node_colors.values()), font_color='white')
    plt.savefig(location, bbox_inches='tight')

