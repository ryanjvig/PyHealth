"""VSAC value set completion: RASCMLP against PyHealth's MLP and retrieval only.

Author: Ryan Vig (NetID: rvig2)
Paper: Retrieve, Then Classify: Corpus-Grounded Automation of Clinical Value
    Set Authoring (Mukherjee et al., 2026)
Paper link: https://arxiv.org/abs/2604.14616

Question: ``examples/vsac_value_set_completion_mlp.py`` found PyHealth's
``MLP`` at pair AUROC 0.730 and value-set F1 0.187, against the paper's MLP
at 0.799 and 0.250, and suggested the architecture as the cause: ``MLP`` runs
one tower per input with no BatchNorm or dropout, while the paper uses one
512-256-64 MLP on the concatenated vector. This script tests that directly.
All models share one ValueSetCompletion task, split and threshold rule, so
only the model differs:

- ``retrieval_only``: every candidate predicted in the set (paper F1 0.136).
- ``pyhealth_mlp``: PyHealth's ``MLP`` with the other example's settings
  (hidden 256, 2 layers).
- ``rasc_mlp``: :class:`~pyhealth.models.RASCMLP`, the paper's architecture.

Ablations of ``rasc_mlp`` (none are in the paper, which reports one MLP
configuration). Each changes one setting:

- ``rasc_no_bn``: no BatchNorm (``batch_norm=False``). Separates the effect
  of BatchNorm from that of concatenating the inputs into one MLP.
- ``rasc_no_dropout`` and ``rasc_dropout_0.5``: dropout 0 and 0.5 (paper 0.3).
- ``rasc_shallow`` and ``rasc_deep``: hidden layers ``(256,)`` and
  ``(1024, 512, 256, 64)`` (paper ``(512, 256, 64)``).
- ``rasc_lr_1e-4`` and ``rasc_lr_1e-3``: learning rate (paper 3e-4).

Every trained model uses class-balanced BCE, AdamW and early stopping on
validation PR-AUC; the threshold maximizes validation pair F1. Test metrics are pair
AUROC and AP, and value-set precision, recall and F1 (mean of per-set F1, the
paper's number, and F1 of the macro precision and recall), overall, on value
sets with full retrieval recall (RR@K = 1), and by value set type. Intervals
are 95% percentile bootstraps over value sets. Differences ``rasc_mlp - X``
for every other trained model X are paired on the same resamples, per seed.

Usage:

    # Smoke test on generated value sets with a stub encoder (under a
    # minute). Its scores carry no signal: the corpus has ~40 test sets and
    # its retrieval score alone is near chance.
    python examples/vsac_value_set_completion_rasc_mlp.py --synthetic

    # Real VSAC download (expansions/ + split_manifest.jsonl), on a GPU.
    python examples/vsac_value_set_completion_rasc_mlp.py --root vsac_corpus \\
        --seeds 0 1 2
    # Only some models:
    python examples/vsac_value_set_completion_rasc_mlp.py --root vsac_corpus \\
        --models retrieval_only pyhealth_mlp rasc_mlp rasc_no_bn

The retrieval index searches only value sets in ``--source-splits`` (default:
the benchmark's train, val and test sets, as in the paper's 11,803-title
index). Results go to ``--output``: ``results.json``, ``results.md``, and
``per_set_<model>_seed<seed>.csv``.

Results: not yet run on VSAC.
"""

import argparse
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import torch

from pyhealth.datasets import VSACDataset, get_dataloader
from pyhealth.metrics import (
    best_f1_threshold,
    bootstrap_ci,
    paired_bootstrap_diff,
    value_set_bootstrap_ci,
    value_set_paired_bootstrap_diff,
    value_set_scores,
    value_set_summary,
)
from pyhealth.models import MLP, RASCMLP
from pyhealth.tasks import ValueSetCompletion, build_retrieval_index
from pyhealth.tasks.value_set_completion import sapbert_encoder
from pyhealth.trainer import Trainer

# The synthetic corpus and stub encoder of the sibling example.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vsac_value_set_completion_mlp import (
    SYNTHETIC_CAP,
    stub_encoder,
    write_synthetic,
)

REFERENCE = "rasc_mlp"
# name -> (RASCMLP keyword arguments, learning rate or None for --lr).
RASC_CONFIGS: dict[str, tuple[dict[str, Any], float | None]] = {
    "rasc_mlp": ({}, None),
    "rasc_no_bn": ({"batch_norm": False}, None),
    "rasc_no_dropout": ({"dropout": 0.0}, None),
    "rasc_dropout_0.5": ({"dropout": 0.5}, None),
    "rasc_shallow": ({"hidden_dims": (256,)}, None),
    "rasc_deep": ({"hidden_dims": (1024, 512, 256, 64)}, None),
    "rasc_lr_1e-4": ({}, 1e-4),
    "rasc_lr_1e-3": ({}, 1e-3),
}
MODELS = ("retrieval_only", "pyhealth_mlp", *RASC_CONFIGS)


def build_model(name: str, train: Any, lr: float) -> tuple[torch.nn.Module, float]:
    """Builds a trainable model and its learning rate.

    Args:
        name: A key of ``MODELS`` other than ``"retrieval_only"``.
        train: The training sample dataset.
        lr: The default learning rate (``--lr``).

    Returns:
        The model and the learning rate to train it with.
    """
    if name == "pyhealth_mlp":
        return MLP(dataset=train, embedding_dim=256, hidden_dim=256, n_layers=2), lr
    kwargs, config_lr = RASC_CONFIGS[name]
    return RASCMLP(train, **kwargs), config_lr or lr


@torch.no_grad()
def predict(model: torch.nn.Module, dataset: Any, batch_size: int) -> pd.DataFrame:
    """Scores every pair, keeping each pair's value set and code.

    Args:
        model: A trained model.
        dataset: A sample dataset from ValueSetCompletion.
        batch_size: Pairs per batch.

    Returns:
        Columns ``set_id``, ``code``, ``n_true_codes``, ``y_true``, ``y_prob``.
    """
    model.eval()
    parts = defaultdict(list)
    for batch in get_dataloader(dataset, batch_size):
        out = model(**batch)
        parts["set_id"] += list(batch["patient_id"])
        parts["code"] += list(batch["code"])
        parts["n_true_codes"] += [int(n) for n in batch["n_true_codes"]]
        parts["y_true"].append(out["y_true"].cpu().numpy().ravel())
        parts["y_prob"].append(out["y_prob"].cpu().numpy().ravel())
    return pd.DataFrame(
        {
            "set_id": parts["set_id"],
            "code": parts["code"],
            "n_true_codes": parts["n_true_codes"],
            "y_true": np.concatenate(parts["y_true"]).astype(int),
            "y_prob": np.concatenate(parts["y_prob"]),
        }
    )


def pairs_only(dataset: Any) -> pd.DataFrame:
    """Labels and identifiers of every pair, for retrieval only.

    Args:
        dataset: A sample dataset from ValueSetCompletion.

    Returns:
        Columns as :func:`predict`, with ``y_prob`` 1 for every pair.
    """
    rows = [
        (s["patient_id"], s["code"], int(s["n_true_codes"]), int(s["label"]))
        for s in dataset
    ]
    frame = pd.DataFrame(rows, columns=["set_id", "code", "n_true_codes", "y_true"])
    return frame.assign(y_prob=1.0)


def score(pairs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Per-value-set scores of predicted pairs.

    Args:
        pairs: Output of :func:`predict` or :func:`pairs_only`.
        threshold: Probability cutoff.

    Returns:
        Output of :func:`~pyhealth.metrics.value_set_scores`.
    """
    return value_set_scores(
        pairs["set_id"],
        pairs["code"],
        pairs["y_true"],
        pairs["y_prob"],
        pairs["n_true_codes"],
        threshold,
    )


def run_model(
    name: str,
    seed: int,
    splits: tuple[Any, Any, Any],
    types: pd.Series,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Trains (if needed) and evaluates one model with one seed.

    Args:
        name: Key of ``MODELS``.
        seed: Seed for weights and shuffling.
        splits: Train, val and test sample datasets.
        types: Value set type by OID.
        args: Parsed command-line arguments.

    Returns:
        The result record, the test pairs with predictions, and per-set
        scores with a ``type`` column.
    """
    train, val, test = splits
    if name == "retrieval_only":
        threshold = 1.0
        test_pairs = pairs_only(test)
    else:
        torch.manual_seed(seed)
        model, lr = build_model(name, train, args.lr)
        trainer = Trainer(
            model,
            metrics=["roc_auc", "pr_auc"],
            device=args.device,
            output_path=str(Path(args.output) / "runs"),
            exp_name=f"{name}_seed{seed}",
        )
        trainer.train(
            train_dataloader=get_dataloader(
                train, args.batch_size, shuffle=True, seed=seed
            ),
            val_dataloader=get_dataloader(val, args.batch_size),
            epochs=args.epochs,
            optimizer_class=torch.optim.AdamW,
            optimizer_params={"lr": lr},
            weight_decay=args.weight_decay,
            monitor="pr_auc",
            patience=args.patience,
        )
        val_pairs = predict(model, val, args.batch_size)
        threshold = best_f1_threshold(val_pairs["y_true"], val_pairs["y_prob"])
        test_pairs = predict(model, test, args.batch_size)

    per_set = score(test_pairs, threshold)
    per_set["type"] = per_set["set_id"].map(types).fillna("Unknown")
    trained = name != "retrieval_only"
    groups = test_pairs["set_id"].to_numpy()
    record: dict[str, Any] = {
        "model": name,
        "seed": seed,
        "threshold": threshold,
        "value_set": value_set_bootstrap_ci(
            per_set, n_boot=args.n_boot, seed=seed
        ),
        "rr_at_k_1": value_set_summary(per_set[per_set["retrieval_recall"] == 1]),
        "by_type": {
            t: value_set_summary(rows) for t, rows in per_set.groupby("type")
        },
    }
    for metric in ("roc_auc", "pr_auc"):
        record[f"pair_{metric}"] = (
            bootstrap_ci(
                test_pairs["y_true"].to_numpy(),
                test_pairs["y_prob"].to_numpy(),
                metric,
                groups=groups,
                n_boot=args.pair_n_boot,
                seed=seed,
            )
            if trained
            else None
        )
    return record, test_pairs, per_set


def compare(
    other: str,
    pairs: dict[str, pd.DataFrame],
    per_set: dict[str, pd.DataFrame],
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Paired bootstrap differences, ``rasc_mlp`` minus another model.

    Args:
        other: The model compared with ``rasc_mlp``.
        pairs: Test pairs with predictions, by model.
        per_set: Per-set scores, by model.
        seed: The seed both models were trained with.
        args: Parsed command-line arguments.

    Returns:
        Value-set and pair-level differences with intervals.
    """
    a, b = pairs[REFERENCE], pairs[other]
    key = ["set_id", "code", "y_true"]
    # Same task cache, so both list the same pairs; check before pairing.
    if not a[key].equals(b[key]):
        raise RuntimeError("Models were scored on different test pairs.")
    groups = a["set_id"].to_numpy()
    return {
        "other": other,
        "seed": seed,
        "value_set": value_set_paired_bootstrap_diff(
            per_set[REFERENCE], per_set[other], n_boot=args.n_boot, seed=seed
        ),
        "pair_roc_auc": paired_bootstrap_diff(
            a["y_true"].to_numpy(),
            a["y_prob"].to_numpy(),
            b["y_prob"].to_numpy(),
            "roc_auc",
            groups=groups,
            n_boot=args.pair_n_boot,
            seed=seed,
        ),
    }


def fmt(ci: dict[str, float] | None) -> str:
    """Formats an interval as ``estimate [lower, upper]``."""
    if ci is None:
        return "n/a"
    return f"{ci['estimate']:.3f} [{ci['lower']:.3f}, {ci['upper']:.3f}]"


def write_report(
    results: list[dict], comparisons: list[dict], meta: dict, output: Path
) -> str:
    """Writes ``results.json`` and ``results.md``; returns the markdown.

    Args:
        results: One :func:`run_model` record per model and seed.
        comparisons: :func:`compare` records.
        meta: Run settings and pair counts.
        output: Output directory.

    Returns:
        The markdown report.
    """
    output.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "results": results, "comparisons": comparisons}
    (output / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    lines = [
        f"Test: {meta['n_pairs']['test']:,} pairs. 95% bootstrap intervals.",
        "",
        (
            "| Model | Seed | Pair AUROC | Pair AP | VS precision | VS recall "
            "| VS F1 (mean of per-set) | F1 of macro P/R | VS F1, RR@K=1 |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        vs = r["value_set"]
        lines.append(
            f"| {r['model']} | {r['seed']} | {fmt(r['pair_roc_auc'])} "
            f"| {fmt(r['pair_pr_auc'])} | {fmt(vs['macro_precision'])} "
            f"| {fmt(vs['macro_recall'])} | {fmt(vs['mean_f1'])} "
            f"| {fmt(vs['f1_of_macro'])} | {r['rr_at_k_1']['mean_f1']:.3f} "
            f"(n={r['rr_at_k_1']['n_value_sets']}) |"
        )
    seeds = sorted({r["seed"] for r in results})
    if len(seeds) > 1:
        lines += ["", "| Model | Mean ± SD over seeds: VS F1 | Pair AUROC |"]
        lines.append("|---|---|---|")
        for name in dict.fromkeys(r["model"] for r in results):
            rows = [r for r in results if r["model"] == name]
            if name == "retrieval_only":
                continue
            f1 = [r["value_set"]["mean_f1"]["estimate"] for r in rows]
            auc = [r["pair_roc_auc"]["estimate"] for r in rows]
            lines.append(
                f"| {name} | {np.mean(f1):.3f} ± {np.std(f1, ddof=1):.3f} "
                f"| {np.mean(auc):.3f} ± {np.std(auc, ddof=1):.3f} |"
            )
    if comparisons:
        lines += [
            "",
            "rasc_mlp - model (paired over the same value sets):",
            "",
            "| Model | Seed | VS F1 | Pair AUROC |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {c['other']} | {c['seed']} | {fmt(c['value_set']['mean_f1'])} "
            f"| {fmt(c['pair_roc_auc'])} |"
            for c in comparisons
        ]
    text = "\n".join(lines) + "\n"
    (output / "results.md").write_text(text)
    return text


def value_set_types(dataset: VSACDataset) -> pd.Series:
    """Returns each value set's type, indexed by OID.

    Args:
        dataset: The VSAC dataset.

    Returns:
        Type label by OID.
    """
    frame = (
        dataset.global_event_df.filter(pl.col("event_type") == "valuesets")
        .select("patient_id", "valuesets/type")
        .collect()
        .to_pandas()
    )
    return frame.set_index("patient_id")["valuesets/type"]


def main() -> None:
    """Parses arguments, builds the index if needed, and runs every model."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", help="VSAC download (expansions/ + manifest)")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--index-dir", help="Reuse or write the retrieval index")
    parser.add_argument("--cache-dir", help="PyHealth cache directory")
    parser.add_argument("--output", default="vsac_rasc_mlp_results")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--source-splits", nargs="+", default=["train", "val", "test"]
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--pair-n-boot", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    unknown = set(args.models) - set(MODELS)
    if unknown or not (args.root or args.synthetic):
        parser.error(f"need --root or --synthetic; known models: {list(MODELS)}")

    work = Path(tempfile.mkdtemp()) if args.synthetic else None
    task_args: dict[str, Any] = {"k": args.k}
    if args.synthetic:
        write_synthetic(work / "vsac", n_topics=20)
        args.root = str(work / "vsac")
        args.cache_dir = str(work / "cache")
        args.index_dir = args.index_dir or str(work / "index")
        args.source_splits = ["train", "val", "test", "corpus"]
        args.epochs = min(args.epochs, 5)
        args.batch_size = min(args.batch_size, 128)
        args.n_boot = min(args.n_boot, 200)
        args.pair_n_boot = min(args.pair_n_boot, 50)
        task_args.update(k=min(args.k, 5), max_codes_per_set=SYNTHETIC_CAP)
    dataset = VSACDataset(root=args.root, cache_dir=args.cache_dir)
    index_dir = args.index_dir or str(Path(args.output) / "index")
    if not (Path(index_dir) / "meta.json").exists():
        build_retrieval_index(
            dataset,
            stub_encoder if args.synthetic else sapbert_encoder(device=args.device),
            index_dir,
            max_k=max(task_args["k"], 10),
            source_splits=args.source_splits,
        )
    task = ValueSetCompletion(index_dir, **task_args)
    splits = dataset.set_task(
        task, num_workers=args.num_workers, split=dataset.manifest_split()
    )
    types = value_set_types(dataset)
    meta = {
        "args": vars(args),
        "task_args": task_args,
        "n_pairs": dict(zip(("train", "val", "test"), map(len, splits))),
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results, comparisons = [], []
    for seed in args.seeds:
        pairs, per_sets = {}, {}
        for name in args.models:
            if name == "retrieval_only" and seed != args.seeds[0]:
                continue  # deterministic
            record, pairs[name], per_sets[name] = run_model(
                name, seed, splits, types, args
            )
            per_set_path = output / f"per_set_{name}_seed{seed}.csv"
            per_sets[name].to_csv(per_set_path, index=False)
            results.append(record)
            write_report(results, comparisons, meta, output)
        if REFERENCE in pairs:
            for other in pairs:
                if other not in (REFERENCE, "retrieval_only"):
                    comparisons.append(compare(other, pairs, per_sets, seed, args))
            write_report(results, comparisons, meta, output)
    print(write_report(results, comparisons, meta, output))
    if work is not None:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()
