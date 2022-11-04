from typing import Any, Dict


def add_prefix_to_metrics(metrics: Dict[str, Any], prefix: str):
    metrics_: Dict[str, Any] = {}
    for k, v in metrics.items():
        metrics_[f"{prefix}/{k}"] = v
    return metrics_
