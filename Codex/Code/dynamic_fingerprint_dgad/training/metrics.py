import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    order = np.argsort(-scores)
    num_pos = int(labels.sum())
    prevalence = float(num_pos / max(len(labels), 1))
    auprc = float(average_precision_score(labels, scores))
    top_pos_k = max(1, num_pos)
    top_01pct_k = max(1, int(np.ceil(0.001 * len(labels))))
    top_05pct_k = max(1, int(np.ceil(0.005 * len(labels))))
    top_1pct_k = max(1, int(np.ceil(0.01 * len(labels))))
    top_pos_hits = int(labels[order[:top_pos_k]].sum())
    top_01pct_hits = int(labels[order[:top_01pct_k]].sum())
    top_05pct_hits = int(labels[order[:top_05pct_k]].sum())
    top_1pct_hits = int(labels[order[:top_1pct_k]].sum())
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": auprc,
        "prevalence": prevalence,
        "normalized_ap_lift": float((auprc - prevalence) / max(1.0 - prevalence, 1e-12)),
        "precision_at_anomaly_count": float(top_pos_hits / top_pos_k),
        "precision_at_0_1pct": float(top_01pct_hits / top_01pct_k),
        "precision_at_0_5pct": float(top_05pct_hits / top_05pct_k),
        "recall_at_anomaly_count": float(top_pos_hits / max(num_pos, 1)),
        "recall_at_0_1pct": float(top_01pct_hits / max(num_pos, 1)),
        "recall_at_0_5pct": float(top_05pct_hits / max(num_pos, 1)),
        "precision_at_1pct": float(top_1pct_hits / top_1pct_k),
        "recall_at_1pct": float(top_1pct_hits / max(num_pos, 1)),
    }
