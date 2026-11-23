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

    # Synthetic corpus and stub encoder, CPU (about 1 minute per seed).
    python examples/vsac_value_set_completion_rasc_mlp.py --synthetic \\
        --seeds 0 1 2

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

Synthetic setup (``--synthetic``, see :func:`write_synthetic`): 180 value
sets (12 topics x 3 aspects x 5 variants; 108 train, 36 val, 36 test),
bag-of-words stub encoder, K=5, 15 epochs at most, patience 3, batch 256,
AdamW (lr 3e-4, weight decay 1e-4), seeds 0-2, CPU. 1,737 test pairs.

Findings on synthetic data (mean +/- SD over 3 seeds):

    Model              VS F1            Pair AUROC
    retrieval_only     0.371            n/a
    pyhealth_mlp       0.333 +/- 0.035  0.553 +/- 0.039
    rasc_mlp           0.416 +/- 0.028  0.717 +/- 0.047
    rasc_no_bn         0.385 +/- 0.028  0.671 +/- 0.087
    rasc_no_dropout    0.426 +/- 0.015  0.728 +/- 0.033
    rasc_dropout_0.5   0.366 +/- 0.029  0.668 +/- 0.042
    rasc_shallow       0.310 +/- 0.036  0.662 +/- 0.033
    rasc_deep          0.423 +/- 0.053  0.740 +/- 0.052
    rasc_lr_1e-4       0.376 +/- 0.022  0.656 +/- 0.038
    rasc_lr_1e-3       0.406 +/- 0.079  0.734 +/- 0.073

- One concatenated MLP beats per-input towers: rasc_mlp exceeds
  pyhealth_mlp in pair AUROC by 0.14-0.18 in every seed (paired 95% CIs
  exclude 0) and in value-set F1 by 0.02-0.13 (CI excludes 0 in 2 of 3
  seeds). pyhealth_mlp falls below retrieval only; rasc_mlp rises above it.
  This matches the direction of the real-data gap reported by the sibling
  example (PyHealth MLP 0.730 / 0.187 vs paper MLP 0.799 / 0.250).
- BatchNorm explains part of the gain: removing it costs 0.03 F1 and 0.05
  AUROC on average, but the paired CI excludes 0 for AUROC in only 2 of 3
  seeds. Most of the gap over pyhealth_mlp comes from the rest of the
  design (one MLP over the concatenated inputs, dropout, layer widths),
  which this script does not separate further.
- Capacity: one hidden layer of 256 (rasc_shallow) is clearly worse (F1
  0.310); a deeper MLP adds nothing over the paper's 512-256-64.
- Dropout 0 and 0.3 are close; 0.5 hurts. lr 1e-4 underfits within 15
  epochs; lr 1e-3 matches on average but is the least stable across seeds.

Caveats: 36 synthetic test sets, a stub encoder and a 15-epoch cap, so
these numbers are not comparable with the paper's and the intervals are
wide. Real VSAC results are pending UMLS access.
"""

import argparse
import json
import random
import shutil
import tempfile
import zlib
from collections import defaultdict
from collections.abc import Sequence
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


SNOMED = "http://snomed.info/sct"
LOINC = "http://loinc.org"
RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"
# Synthetic aspect -> (title/display word, code system, value set type).
ASPECTS = (
    ("disorder", SNOMED, "Condition/Clinical"),
    ("lab", LOINC, "Lab/Observation"),
    ("medication", RXNORM, "Medication"),
)
TOPICS = (
    "asthma", "diabetes", "hypertension", "sepsis", "stroke", "anemia",
    "migraine", "obesity", "pneumonia", "hepatitis", "lupus", "gout",
)
SPLITS = ("train", "train", "train", "val", "test")


def stub_encoder(texts: Sequence[str]) -> np.ndarray:
    """Unit-norm bag-of-words hashing encoder; stands in for SapBERT.

    Texts that share words get higher cosine similarity, which is all the
    synthetic corpus relies on.

    Args:
        texts: Texts to encode.

    Returns:
        A float32 array of shape ``(len(texts), 64)``.
    """
    vecs = np.zeros((len(texts), 64), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in text.lower().split():
            vecs[i, zlib.crc32(word.encode()) % 64] += 1.0
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1.0, norms)


def write_synthetic(root: Path, seed: int = 0) -> None:
    """Writes a synthetic VSAC download whose signals mimic the real task.

    Each value set covers one topic (e.g. asthma) and one aspect (disorder,
    lab or medication, each with its own code system), and draws most codes
    from that topic-aspect's pool of 30. Titles are ``"<topic> <aspect>
    <unique word>"`` and displays ``"<topic> <aspect> <unique word>"``, so:

    - Sets on the same topic and aspect share two title words and get the
      top retrieval scores; same-topic sets of other aspects share one word
      and contribute candidates that are mostly negatives.
    - A true candidate's display shares the aspect word with the title.
    - The code system matches the aspect.

    One set in five also has 1-3 codes from another aspect of its topic, so
    no single signal separates the classes. 12 topics x 3 aspects x 5
    variants give 180 value sets: 108 train, 36 val, 36 test.

    Args:
        root: Output directory; gets ``expansions/`` and
            ``split_manifest.jsonl``.
        seed: Random seed.
    """
    rng = random.Random(seed)
    exp_dir = root / "expansions"
    exp_dir.mkdir(parents=True)
    manifest = []
    for t, topic in enumerate(TOPICS):
        pools = [
            [(f"{t}{a}{c:02d}", f"{topic} {word} w{t}x{a}x{c}") for c in range(30)]
            for a, (word, _, _) in enumerate(ASPECTS)
        ]
        for a, (word, system, value_set_type) in enumerate(ASPECTS):
            for v, split in enumerate(SPLITS):
                oid = f"9.{t}.{a}.{v}-20240101"
                drawn = rng.sample(pools[a], rng.randint(8, 20))
                members = [(system, *m) for m in drawn]
                if rng.random() < 0.2:
                    b = rng.choice([x for x in range(len(ASPECTS)) if x != a])
                    extra = rng.sample(pools[b], rng.randint(1, 3))
                    members += [(ASPECTS[b][1], *m) for m in extra]
                resource = {
                    "resourceType": "ValueSet",
                    "id": oid,
                    "title": f"{topic} {word} v{t}x{a}x{v}",
                    "version": "20240101",
                    "expansion": {
                        "total": len(members),
                        "contains": [
                            {"system": sys_uri, "code": code, "display": display}
                            for sys_uri, code, display in members
                        ],
                    },
                }
                (exp_dir / f"{oid}.json").write_text(json.dumps(resource))
                manifest.append(
                    {
                        "oid": oid,
                        "split": split,
                        "type": value_set_type,
                        "publisher": "Synthetic Steward",
                    }
                )
    (root / "split_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in manifest)
    )


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
        write_synthetic(work / "vsac")
        args.root = str(work / "vsac")
        args.cache_dir = str(work / "cache")
        args.index_dir = args.index_dir or str(work / "index")
        args.epochs = min(args.epochs, 15)
        args.patience = min(args.patience, 3)
        args.batch_size = min(args.batch_size, 256)
        args.n_boot = min(args.n_boot, 200)
        args.pair_n_boot = min(args.pair_n_boot, 50)
        task_args.update(k=min(args.k, 5))
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
