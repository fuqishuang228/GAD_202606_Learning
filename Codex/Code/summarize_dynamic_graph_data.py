#!/usr/bin/env python3
"""Summarize downloaded dynamic graph datasets."""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional


ROOT = Path("/home/qfu/bx82_scratch/qfu/[A]GAD_202606_learning")
DATA_DIR = ROOT / "Data" / "Dynamic_Graph_data"
DOC_DIR = ROOT / "Codex" / "Doc"


def count_lines(path: Path, skip_header: bool = False) -> int:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        total = sum(1 for _ in f)
    return max(0, total - int(skip_header))


def file_size(path: Path) -> str:
    size = path.stat().st_size
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size} B"


def label_counter_str(counter: Counter) -> str:
    return ", ".join(f"{k}: {v}" for k, v in sorted(counter.items(), key=lambda x: str(x[0])))


def summarize_jodie_csv(path: Path) -> Dict[str, object]:
    users = set()
    items = set()
    labels: Counter = Counter()
    feature_dim: Optional[int] = None
    min_ts: Optional[float] = None
    max_ts: Optional[float] = None
    rows = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row:
                continue
            rows += 1
            users.add(row[0])
            items.add(row[1])
            ts = float(row[2])
            min_ts = ts if min_ts is None else min(min_ts, ts)
            max_ts = ts if max_ts is None else max(max_ts, ts)
            labels[row[3]] += 1
            if feature_dim is None:
                feature_dim = len(row) - 4
    return {
        "file": str(path),
        "size": file_size(path),
        "rows": rows,
        "users": len(users),
        "items": len(items),
        "feature_dim": feature_dim,
        "timestamp_min": min_ts,
        "timestamp_max": max_ts,
        "label_counts": dict(labels),
        "header": header,
    }


def summarize_mooc(path: Path) -> Dict[str, object]:
    actions_path = path / "act-mooc" / "mooc_actions.tsv"
    features_path = path / "act-mooc" / "mooc_action_features.tsv"
    labels_path = path / "act-mooc" / "mooc_action_labels.tsv"
    users = set()
    targets = set()
    min_ts: Optional[int] = None
    max_ts: Optional[int] = None
    rows = 0
    with actions_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 4 or row[0] == "ACTIONID":
                continue
            rows += 1
            users.add(row[1])
            targets.add(row[2])
            ts = int(float(row[3]))
            min_ts = ts if min_ts is None else min(min_ts, ts)
            max_ts = ts if max_ts is None else max(max_ts, ts)

    labels: Counter = Counter()
    with labels_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2 or row[0] == "ACTIONID":
                continue
            labels[row[1]] += 1

    feature_dim: Optional[int] = None
    with features_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) > 1 and row[0] != "ACTIONID":
                feature_dim = len(row) - 1
                break

    return {
        "file": str(actions_path),
        "archive": str(path / "act-mooc.tar.gz"),
        "size": file_size(path / "act-mooc.tar.gz"),
        "rows": rows,
        "users": len(users),
        "targets": len(targets),
        "feature_dim": feature_dim,
        "timestamp_min": min_ts,
        "timestamp_max": max_ts,
        "label_counts": dict(labels),
    }


def summarize_bitcoin_alpha(path: Path) -> Dict[str, object]:
    csv_path = path / "soc-sign-bitcoinalpha.csv"
    nodes = set()
    ratings: Counter = Counter()
    min_time: Optional[int] = None
    max_time: Optional[int] = None
    rows = 0
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            rows += 1
            nodes.add(row[0])
            nodes.add(row[1])
            rating = int(row[2])
            ratings[rating] += 1
            ts = int(row[3])
            min_time = ts if min_time is None else min(min_time, ts)
            max_time = ts if max_time is None else max(max_time, ts)
    return {
        "file": str(csv_path),
        "archive": str(path / "soc-sign-bitcoinalpha.csv.gz"),
        "size": file_size(path / "soc-sign-bitcoinalpha.csv.gz"),
        "rows": rows,
        "nodes": len(nodes),
        "rating_counts": dict(ratings),
        "positive_edges": sum(v for k, v in ratings.items() if k > 0),
        "negative_edges": sum(v for k, v in ratings.items() if k < 0),
        "timestamp_min": min_time,
        "timestamp_max": max_time,
    }


def summarize_elliptic(path: Path) -> Dict[str, object]:
    features_path = path / "elliptic_txs_features.csv"
    classes_path = path / "elliptic_txs_classes.csv"
    edges_path = path / "elliptic_txs_edgelist.csv"

    tx_ids = set()
    timesteps: Counter = Counter()
    feature_dim: Optional[int] = None
    with features_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            tx_ids.add(row[0])
            timesteps[row[1]] += 1
            if feature_dim is None:
                feature_dim = len(row) - 1

    classes: Counter = Counter()
    with classes_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            classes[row[1]] += 1

    edges = count_lines(edges_path, skip_header=True)
    return {
        "files": [str(features_path), str(classes_path), str(edges_path)],
        "sizes": {
            features_path.name: file_size(features_path),
            classes_path.name: file_size(classes_path),
            edges_path.name: file_size(edges_path),
        },
        "nodes": len(tx_ids),
        "edges": edges,
        "feature_dim_including_time": feature_dim,
        "time_steps": len(timesteps),
        "time_step_min": min(map(int, timesteps.keys())),
        "time_step_max": max(map(int, timesteps.keys())),
        "class_counts": dict(classes),
    }


def markdown_report(stats: Dict[str, Dict[str, object]]) -> str:
    lines = [
        "# Dynamic Graph Dataset Download and Summary",
        "",
        "Base data directory:",
        f"`{DATA_DIR}`",
        "",
        "Generated by:",
        f"`{ROOT / 'Codex' / 'Code' / 'summarize_dynamic_graph_data.py'}`",
        "",
        "## Download Status",
        "",
        "| Dataset | Status | Local folder | Notes |",
        "|---|---:|---|---|",
        f"| EllipticBitcoin | downloaded | `{DATA_DIR / 'EllipticBitcoin'}` | Original requested source is Kaggle; downloaded from a public Hugging Face mirror because this machine has no Kaggle CLI/token. |",
        f"| MOOC | downloaded | `{DATA_DIR / 'MOOC'}` | SNAP `act-mooc.tar.gz`, extracted. |",
        f"| Wiki | downloaded | `{DATA_DIR / 'Wiki'}` | JODIE `wikipedia.csv`. |",
        f"| Reddit | downloaded | `{DATA_DIR / 'Reddit'}` | JODIE `reddit.csv`. |",
        f"| Bitcoin-Alpha | downloaded | `{DATA_DIR / 'Bitcoin-Alpha'}` | SNAP `.csv.gz`, also decompressed. |",
        f"| DGraphFin | manual / pending | `{DATA_DIR / 'DGraphFin'}` | Site requires JS/manual download. Put `DGraphFin.zip` here or in `DGraphFin/raw/` for PyG. |",
        "",
        "## Dataset Information",
        "",
    ]

    ell = stats["EllipticBitcoin"]
    lines += [
        "### EllipticBitcoin",
        "",
        "- Nodes: Bitcoin transactions.",
        "- Edges: directed Bitcoin flow between transactions.",
        "- Edge features: none in the downloaded edge list.",
        "- Node features: 166 values per node if counting the time-step column; commonly used as 165 numeric node features after separating time.",
        "- Time information: discrete time step 1-49 stored on nodes.",
        "- Label information: node labels `1` illicit, `2` licit, `unknown` unlabeled.",
        "- Has anomaly labels: yes, illicit transaction nodes are anomaly/fraud labels.",
        f"- Local stats: nodes={ell['nodes']}, edges={ell['edges']}, time_steps={ell['time_steps']} ({ell['time_step_min']}-{ell['time_step_max']}), labels={ell['class_counts']}.",
        "",
    ]

    mooc = stats["MOOC"]
    lines += [
        "### MOOC",
        "",
        "- Nodes: users and course activities/targets.",
        "- Edges: directed temporal user actions on target activities.",
        "- Edge features: 4 action-level features in `mooc_action_features.tsv`.",
        "- Time information: action timestamp in seconds from an anonymized zero point.",
        "- Label information: action label where 1 means the user drops out after this action; 0 otherwise.",
        "- Has anomaly labels: not a fraud/anomaly dataset by default; labels are dropout/action-state labels.",
        f"- Local stats: users={mooc['users']}, targets={mooc['targets']}, actions={mooc['rows']}, feature_dim={mooc['feature_dim']}, labels={mooc['label_counts']}.",
        "",
    ]

    wiki = stats["Wiki"]
    lines += [
        "### Wiki",
        "",
        "- Nodes: Wikipedia users and pages/items.",
        "- Edges: temporal user-page interactions/edits in JODIE format.",
        f"- Edge features: {wiki['feature_dim']} interaction features.",
        "- Time information: continuous timestamp column, standardized to start at 0.",
        "- Label information: `state_label` on interactions.",
        "- Has anomaly labels: not a fraud/anomaly dataset by default; `state_label` is an interaction state label used by JODIE.",
        f"- Local stats: users={wiki['users']}, items={wiki['items']}, interactions={wiki['rows']}, timestamps={wiki['timestamp_min']}-{wiki['timestamp_max']}, labels={wiki['label_counts']}.",
        "",
    ]

    reddit = stats["Reddit"]
    lines += [
        "### Reddit",
        "",
        "- Nodes: Reddit users and subreddits/items.",
        "- Edges: temporal user-subreddit interactions/posts in JODIE format.",
        f"- Edge features: {reddit['feature_dim']} interaction features.",
        "- Time information: continuous timestamp column, standardized to start at 0.",
        "- Label information: `state_label` on interactions.",
        "- Has anomaly labels: not a fraud/anomaly dataset by default; `state_label` is an interaction state label used by JODIE.",
        f"- Local stats: users={reddit['users']}, items={reddit['items']}, interactions={reddit['rows']}, timestamps={reddit['timestamp_min']}-{reddit['timestamp_max']}, labels={reddit['label_counts']}.",
        "",
    ]

    btc = stats["Bitcoin-Alpha"]
    lines += [
        "### Bitcoin-Alpha",
        "",
        "- Nodes: Bitcoin Alpha users/traders.",
        "- Edges: directed trust ratings from rater to ratee.",
        "- Edge features: rating score from -10 to +10; no separate node features.",
        "- Time information: Unix epoch timestamp per rating edge.",
        "- Label information: signed/weighted edge rating.",
        "- Has anomaly labels: no explicit anomaly labels; negative trust ratings can be treated as suspicious/negative edges for some tasks but are not ground-truth anomalies.",
        f"- Local stats: nodes={btc['nodes']}, edges={btc['rows']}, positive_edges={btc['positive_edges']}, negative_edges={btc['negative_edges']}, timestamps={btc['timestamp_min']}-{btc['timestamp_max']}.",
        "",
    ]

    lines += [
        "### DGraphFin",
        "",
        "- Nodes: fintech platform users.",
        "- Edges: directed social/financial-platform relations between users; PyG exposes `edge_type` and `edge_time`.",
        "- Edge features: anonymized edge type plus edge timestamp/time mark in the PyG object; no downloaded raw file available yet.",
        "- Node features: 17 anonymized user features.",
        "- Time information: edge time (`edge_time`) after loading through PyG.",
        "- Label information: node fraud/normal labels with train/validation/test masks.",
        "- Has anomaly labels: yes, fraudulent user nodes are anomaly labels.",
        "- Local status: not downloaded; manual website/PyG raw zip step is still required.",
        "",
        "Manual DGraphFin placement note:",
        "",
        "```bash",
        f"mkdir -p '{DATA_DIR / 'DGraphFin' / 'raw'}'",
        f"# Download DGraphFin.zip from https://dgraph.xinye.com/ and place it at:",
        f"# {DATA_DIR / 'DGraphFin' / 'raw' / 'DGraphFin.zip'}",
        "```",
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    stats = {
        "EllipticBitcoin": summarize_elliptic(DATA_DIR / "EllipticBitcoin"),
        "MOOC": summarize_mooc(DATA_DIR / "MOOC"),
        "Wiki": summarize_jodie_csv(DATA_DIR / "Wiki" / "wikipedia.csv"),
        "Reddit": summarize_jodie_csv(DATA_DIR / "Reddit" / "reddit.csv"),
        "Bitcoin-Alpha": summarize_bitcoin_alpha(DATA_DIR / "Bitcoin-Alpha"),
    }
    (DOC_DIR / "dynamic_graph_dataset_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (DOC_DIR / "dynamic_graph_dataset_summary.md").write_text(
        markdown_report(stats),
        encoding="utf-8",
    )
    print(f"Wrote {DOC_DIR / 'dynamic_graph_dataset_summary.md'}")
    print(f"Wrote {DOC_DIR / 'dynamic_graph_dataset_stats.json'}")


if __name__ == "__main__":
    main()
