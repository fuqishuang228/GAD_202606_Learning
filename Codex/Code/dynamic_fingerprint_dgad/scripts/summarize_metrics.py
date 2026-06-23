import json
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dynamic_fingerprint_dgad/results/first_full_gpu/metrics.json")
    with open(path) as f:
        obj = json.load(f)
    print(f"metrics: {path}")
    for name, metrics in obj["target"].items():
        auroc = metrics["auroc"]
        auprc = metrics["auprc"]
        print(
            f"{name:12s} rows={metrics['rows']:7d} anomalies={metrics['anomalies']:6d} "
            f"AUROC={auroc:.4f} AUPRC={auprc:.4f} "
            f"score_mean={metrics['score_mean']:.4f} score_std={metrics['score_std']:.4f}"
        )


if __name__ == "__main__":
    main()

