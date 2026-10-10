"""Aggregate the prediction files in missci/predictions/classify-given-gold-premise-raw into results/.

Usage:
    python aggregate_predictions.py                     # scores, summary, per_class and confusion.csv
    python aggregate_predictions.py --eval-set mafalda  # print one table, write nothing
    python aggregate_predictions.py --per-class         # per_class.csv and confusion.csv only
"""

import csv
import glob as globlib
import os
import re
import statistics
from collections import Counter
from functools import cache
from pathlib import Path

import numpy as np
import typer
from sklearn.metrics import f1_score

from analyze_predictions import UNPARSEABLE_CLASS
from analyze_predictions import get_bootstrap_ci
from analyze_predictions import get_mcnemar_test
from analyze_predictions import score_predictions

PREDICTIONS_DIR = "missci/predictions/classify-given-gold-premise-raw"
RESULTS_DIR = "results"
TEMPLATE = "cls_with_premise_classify-D"
# By model size.
MODELS = (
    "gemma-3-4b-it-4bit",
    "Llama-3.1-8B-Instruct-4bit",
    "llama2-13b-qnt4bit",
    "phi-4-4bit",
    "Mistral-Small-3.2-24B-Instruct-2506-4bit",
)
DISPLAY_NAMES = {
    "gemma-3-4b-it-4bit": "Gemma 3 4B",
    "Llama-3.1-8B-Instruct-4bit": "LLaMA 3.1 8B",
    "llama2-13b-qnt4bit": "LLaMA 2 13B-chat",
    "phi-4-4bit": "Phi-4",
    "Mistral-Small-3.2-24B-Instruct-2506-4bit": "Mistral Small 3.2 24B",
}
# MISSCI test, then the cross-dataset sets; the selected configuration, then the ablations.
EVAL_SETS = ("test", "mafalda", "logic")
FULL_DATASET = "o4-mini-k-15-m-5"
DATASETS = (
    FULL_DATASET,
    "o4-mini-k-15-m-5-no-retrieval",
    "o4-mini-k-15-m-5-random-source",
    "o4-mini-k-15-m-0",
    "o4-mini-k-15-m-5-random-content",
    "gold-dev-only",
    "gpt-5-k-15-m-5",
)
# Per-class rows averaged over the models of a table.
ALL_MODELS = "all"
ALL_MODELS_MEAN_COLUMNS = (
    "base_precision", "base_recall", "base_f1", "base_predicted_share",
    "ft_precision_mean", "ft_recall_mean", "ft_f1_mean", "ft_predicted_share_mean", "delta_f1",
)  # fmt: skip
PAIRED_BOOTSTRAP_RESAMPLES = 3000
FILE_PATTERN = re.compile(
    rf"^(?P<model>.+?)_{TEMPLATE}_(?P<eval_set>{'|'.join(EVAL_SETS)})(?:_(?P<dataset>.+)_seed(?P<seed>\d+))?\.jsonl$"
)
SEED_PATTERN = re.compile(r"_seed(\d+)\.jsonl$")


@cache
def score(prediction_file: str) -> dict:
    return score_predictions(prediction_file)


def get_prediction_files(predictions_dir: str) -> list[dict]:
    """Base and fine-tuned prediction files of MODELS, in table order."""
    files = []
    for path in globlib.glob(f"{predictions_dir}/*.jsonl"):
        match = FILE_PATTERN.match(Path(path).name)
        if match and match["model"] in MODELS:
            files.append({"path": path, **match.groupdict()})

    def order(file: dict) -> tuple:
        dataset = file["dataset"] or ""
        return (
            EVAL_SETS.index(file["eval_set"]),
            DATASETS.index(dataset) + 1 if dataset in DATASETS else len(DATASETS) + 1,
            dataset,
            MODELS.index(file["model"]),
            int(file["seed"] or -1),
        )

    return sorted(files, key=order)


def get_seed_files(predictions_dir: str, model: str, eval_set: str, dataset: str) -> dict[int, str]:
    """Fine-tuned prediction files of one model, keyed by seed."""
    pattern = f"{predictions_dir}/{model}_{TEMPLATE}_{eval_set}_{dataset}_seed*.jsonl"
    files = {int(SEED_PATTERN.search(path).group(1)): path for path in globlib.glob(pattern)}
    return dict(sorted(files.items()))


def get_gold_class_macro_f1(report: dict) -> float:
    """Macro F1 over the gold classes only, without ``unk``."""
    f1_scores = [
        scores["f1-score"]
        for label, scores in report.items()
        if isinstance(scores, dict) and label not in ("macro avg", "weighted avg") and scores["support"] > 0
    ]
    return sum(f1_scores) / len(f1_scores)


def get_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else float("nan")


def get_macro_f1(gold: np.ndarray, scored: np.ndarray) -> float:
    """MISSCI's macro F1 (``unk`` averaged in), as in analyze_predictions.py."""
    return f1_score(gold, scored, average="macro", zero_division=0, labels=sorted(set(gold) | set(scored)))


@cache
def get_bootstrap_indices(n_items: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(0)
    return tuple(rng.integers(0, n_items, n_items) for _ in range(PAIRED_BOOTSTRAP_RESAMPLES))


@cache
def get_bootstrap_seed_means(prediction_files: tuple[str, ...]) -> np.ndarray:
    """Seed-mean accuracy and macro F1 on every paired-bootstrap resample of the test items."""
    seeds = [score(path) for path in prediction_files]
    gold = np.array(seeds[0]["gold"])
    scored = [np.array(seed["scored"]) for seed in seeds]
    return np.array(
        [
            (
                np.mean([(s[idx] == gold[idx]).mean() for s in scored]),
                np.mean([get_macro_f1(gold[idx], s[idx]) for s in scored]),
            )
            for idx in get_bootstrap_indices(len(gold))
        ]
    )


def get_macro_f1_ci(prediction_files: tuple[str, ...]) -> tuple[float, float]:
    """Percentile 95% CI of the seed-mean macro F1 over the paired-bootstrap resamples."""
    resamples = get_bootstrap_seed_means(prediction_files)[:, 1]
    return float(np.percentile(resamples, 2.5)), float(np.percentile(resamples, 97.5))


def compare_with_full(
    row: dict, full_files: dict[int, str], ablation_files: dict[int, str], item_ids: list[str]
) -> None:
    """Add the ``vs_full_*`` columns: the ablation's seeds against the full set's seeds."""
    if not set(ablation_files) <= set(full_files):
        return
    for seed in ablation_files:
        if score(full_files[seed])["ids"] != item_ids:
            message = f"{full_files[seed]} does not contain the same items in the same order as the ablation files"
            raise ValueError(message)
    tests = [
        get_mcnemar_test(score(full_files[seed])["correct"], score(path)["correct"])
        for seed, path in ablation_files.items()
    ]
    row["vs_full_full_only_correct"] = [full_only for _, full_only, _ in tests]
    row["vs_full_ablation_only_correct"] = [ablation_only for ablation_only, _, _ in tests]
    row["vs_full_mcnemar_p"] = [p_value for _, _, p_value in tests]
    if list(ablation_files) != list(full_files):
        return
    full = tuple(full_files.values())
    differences = get_bootstrap_seed_means(tuple(ablation_files.values())) - get_bootstrap_seed_means(full)
    for i, metric in enumerate(("acc", "macro_f1")):
        resamples = differences[:, i]
        full_mean = statistics.mean(score(path)[metric] for path in full)
        row[f"vs_full_delta_{metric}"] = row[f"ft_{metric}_mean"] - full_mean
        row[f"vs_full_delta_{metric}_ci_low"] = float(np.percentile(resamples, 2.5))
        row[f"vs_full_delta_{metric}_ci_high"] = float(np.percentile(resamples, 97.5))
        row[f"vs_full_delta_{metric}_p"] = float(min(1.0, 2 * min((resamples <= 0).mean(), (resamples >= 0).mean())))


def add_answered_items(row: dict, base: dict, seeds: list[dict]) -> None:
    """Base refusals and base vs fine-tuned accuracy on the items the base model answered."""
    answered = [i for i, predicted in enumerate(base["predicted"]) if predicted != UNPARSEABLE_CLASS]
    row["base_refusals"] = base["n"] - len(answered)
    if not answered:
        return
    base_correct = [base["correct"][i] for i in answered]
    row["answered_base_acc"] = sum(base_correct) / len(answered)
    seed_correct = [[seed["correct"][i] for i in answered] for seed in seeds]
    row["answered_seed_acc"] = [sum(correct) / len(answered) for correct in seed_correct]
    tests = [get_mcnemar_test(base_correct, correct) for correct in seed_correct]
    row["answered_ft_only_correct"] = [ft_only for ft_only, _, _ in tests]
    row["answered_base_only_correct"] = [base_only for _, base_only, _ in tests]
    row["answered_mcnemar_p"] = [p_value for _, _, p_value in tests]


def aggregate_model(
    predictions_dir: str, model: str, eval_set: str, dataset: str, bootstrap_resamples: int
) -> dict | None:
    base_file = f"{predictions_dir}/{model}_{TEMPLATE}_{eval_set}.jsonl"
    seed_files = get_seed_files(predictions_dir, model, eval_set, dataset)
    has_base = Path(base_file).exists()
    if not has_base and not seed_files:
        print(f"skip {model}: no base file and no seed files")
        return None

    files = ([base_file] if has_base else []) + list(seed_files.values())
    item_ids = score(files[0])["ids"]
    for path in files[1:]:
        if score(path)["ids"] != item_ids:
            message = f"{path} does not contain the same items in the same order as {files[0]}"
            raise ValueError(message)

    row: dict = {"model": model, "eval_set": eval_set, "dataset": dataset, "n_items": len(item_ids)}
    base = score(base_file) if has_base else None
    if base:
        row["base_acc"] = base["acc"]
        row["base_acc_ci_low"], row["base_acc_ci_high"] = get_bootstrap_ci(base["correct"], bootstrap_resamples)
        row["base_macro_f1"] = base["macro_f1"]
        row["base_macro_f1_ci_low"], row["base_macro_f1_ci_high"] = get_macro_f1_ci((base_file,))
        row["base_macro_f1_gold"] = get_gold_class_macro_f1(base["report"])
        row["base_unk"] = base["unk"]

    seeds = [score(path) for path in seed_files.values()]
    row["seeds"] = list(seed_files)
    if seeds:
        accs = [seed["acc"] for seed in seeds]
        f1s = [seed["macro_f1"] for seed in seeds]
        gold_f1s = [get_gold_class_macro_f1(seed["report"]) for seed in seeds]
        mean_correct = [sum(values) / len(seeds) for values in zip(*(seed["correct"] for seed in seeds), strict=True)]
        row["ft_acc_mean"], row["ft_acc_std"] = statistics.mean(accs), get_std(accs)
        row["ft_acc_ci_low"], row["ft_acc_ci_high"] = get_bootstrap_ci(mean_correct, bootstrap_resamples)
        row["ft_macro_f1_mean"], row["ft_macro_f1_std"] = statistics.mean(f1s), get_std(f1s)
        row["ft_macro_f1_ci_low"], row["ft_macro_f1_ci_high"] = get_macro_f1_ci(tuple(seed_files.values()))
        row["ft_macro_f1_gold_mean"], row["ft_macro_f1_gold_std"] = statistics.mean(gold_f1s), get_std(gold_f1s)
        row["seed_acc"], row["seed_macro_f1"], row["seed_macro_f1_gold"] = accs, f1s, gold_f1s
        row["seed_unk"] = [seed["unk"] for seed in seeds]
    if base and seeds:
        row["delta_acc"] = row["ft_acc_mean"] - base["acc"]
        row["delta_macro_f1"] = row["ft_macro_f1_mean"] - base["macro_f1"]
        tests = [get_mcnemar_test(base["correct"], seed["correct"]) for seed in seeds]
        row["ft_only_correct"] = [ft_only for ft_only, _, _ in tests]
        row["base_only_correct"] = [base_only for _, base_only, _ in tests]
        row["mcnemar_p"] = [p_value for _, _, p_value in tests]
        add_answered_items(row, base, seeds)
    if seeds and dataset != FULL_DATASET:
        full_files = get_seed_files(predictions_dir, model, eval_set, FULL_DATASET)
        compare_with_full(row, full_files, seed_files, item_ids)
    return row


def get_label_scores(result: dict, label: str) -> dict:
    """Precision, recall, F1 and share of predictions of one label of a scored file (0 if never seen)."""
    if label == "accuracy":
        return {"f1": result["acc"]}
    scores = result["report"].get(label, {})
    label_scores = {
        "precision": scores.get("precision", 0.0),
        "recall": scores.get("recall", 0.0),
        "f1": scores.get("f1-score", 0.0),
    }
    if label != "macro avg":
        label_scores["predicted_share"] = result["scored"].count(label) / result["n"]
    return label_scores


def get_confusion_rows(results: list[dict], row: dict) -> list[dict]:
    """Non-empty (gold, scored) cells of the scored files pooled, with their share of the gold class."""
    cells = Counter(pair for result in results for pair in zip(result["gold"], result["scored"], strict=True))
    gold_counts = Counter(gold for result in results for gold in result["gold"])
    return [
        {**row, "gold": gold, "predicted": predicted, "count": count, "share": count / gold_counts[gold]}
        for (gold, predicted), count in sorted(cells.items())
    ]


def aggregate_per_class(
    predictions_dir: str, models: list[str] | tuple[str, ...], eval_set: str, dataset: str
) -> tuple[list[dict], list[dict]]:
    """Per-class rows and confusion cells of one table, per model and over all models."""
    rows, cells, model_rows = [], [], {}
    all_base, all_seeds = [], []
    for model in models:
        base_file = f"{predictions_dir}/{model}_{TEMPLATE}_{eval_set}.jsonl"
        seed_files = get_seed_files(predictions_dir, model, eval_set, dataset)
        if not Path(base_file).exists() or not seed_files:
            continue
        base = score(base_file)
        seeds = [score(path) for path in seed_files.values()]
        table = {"eval_set": eval_set, "dataset": dataset, "model": model}
        for label in [*sorted(set(base["gold"])), "unk", "macro avg", "accuracy"]:
            row = {**table, "label": label, "seeds": list(seed_files)}
            if label in ("macro avg", "accuracy"):
                row["support"] = base["n"]
            else:
                row["support"] = base["gold"].count(label)
                row["gold_share"] = row["support"] / base["n"]
            row |= {f"base_{key}": value for key, value in get_label_scores(base, label).items()}
            seed_scores = [get_label_scores(seed, label) for seed in seeds]
            row |= {f"ft_{key}_mean": statistics.mean(s[key] for s in seed_scores) for key in seed_scores[0]}
            row["ft_f1_std"] = get_std([s["f1"] for s in seed_scores])
            row["delta_f1"] = row["ft_f1_mean"] - row["base_f1"]
            rows.append(row)
            model_rows.setdefault(label, []).append(row)
        cells += get_confusion_rows([base], {**table, "setting": "base"})
        cells += get_confusion_rows(seeds, {**table, "setting": "fine-tuned"})
        all_base.append(base)
        all_seeds += seeds
    if not rows:
        return [], []

    # Means over the models, each model's seeds weighing as one model.
    table = {"eval_set": eval_set, "dataset": dataset, "model": ALL_MODELS}
    for label, label_rows in model_rows.items():
        row = {**table, "label": label, "support": label_rows[0]["support"], "n_models": len(label_rows)}
        if "gold_share" in label_rows[0]:
            row["gold_share"] = label_rows[0]["gold_share"]
        for key in ALL_MODELS_MEAN_COLUMNS:
            if key in label_rows[0]:
                row[key] = statistics.mean(label_row[key] for label_row in label_rows)
        row["models_improved"] = sum(label_row["delta_f1"] > 0 for label_row in label_rows)
        rows.append(row)
    cells += get_confusion_rows(all_base, {**table, "setting": "base"})
    cells += get_confusion_rows(all_seeds, {**table, "setting": "fine-tuned"})
    return rows, cells


def aggregate_per_class_tables(
    tables: list[tuple[str, str]], models: list[str] | tuple[str, ...], predictions_dir: str, show: bool
) -> tuple[list[dict], list[dict]]:
    """Per-class rows and confusion cells of every table among ``tables``."""
    rows, cells = [], []
    for table_eval_set, table_dataset in tables:
        table_rows, table_cells = aggregate_per_class(predictions_dir, models, table_eval_set, table_dataset)
        if table_rows and show:
            print(f"\n=== eval set {table_eval_set}, fine-tuning dataset {table_dataset}, per class")
            print_per_class(table_rows, table_cells)
        rows += table_rows
        cells += table_cells
    return rows, cells


def format_mean_std(row: dict, key: str) -> str:
    if f"{key}_mean" not in row:
        return "-"
    if len(row["seeds"]) == 1:
        return f"{row[f'{key}_mean']:.3f} (1 seed)"
    return f"{row[f'{key}_mean']:.3f} ± {row[f'{key}_std']:.3f}"


def print_summary(rows: list[dict]) -> None:
    header = (
        f"{'model':<22} {'base acc':>8} {'base 95% CI':>15} {'base F1':>7} {'F1 95% CI':>15} {'(gold)':>7} | "
        f"{'seeds':>5} {'FT acc':>13} {'FT 95% CI':>15} {'FT F1':>13} {'FT F1 95% CI':>15} {'FT F1 (gold)':>13} | "
        f"{'Δacc':>6} {'ΔF1':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        name = DISPLAY_NAMES.get(row["model"], row["model"])
        base = (
            f"{row['base_acc']:>8.3f} [{row['base_acc_ci_low']:.3f}, {row['base_acc_ci_high']:.3f}] "
            f"{row['base_macro_f1']:>7.3f} [{row['base_macro_f1_ci_low']:.3f}, {row['base_macro_f1_ci_high']:.3f}] "
            f"{row['base_macro_f1_gold']:>7.3f}"
            if "base_acc" in row
            else f"{'-':>8} {'-':>15} {'-':>7} {'-':>15} {'-':>7}"
        )
        ft_ci = f"[{row['ft_acc_ci_low']:.3f}, {row['ft_acc_ci_high']:.3f}]" if "ft_acc_mean" in row else "-"
        ft_f1_ci = (
            f"[{row['ft_macro_f1_ci_low']:.3f}, {row['ft_macro_f1_ci_high']:.3f}]" if "ft_macro_f1_mean" in row else "-"
        )
        delta = f"{row['delta_acc']:>+6.3f} {row['delta_macro_f1']:>+6.3f}" if "delta_acc" in row else ""
        print(
            f"{name[:22]:<22} {base} | {len(row['seeds']):>5} {format_mean_std(row, 'ft_acc'):>13} {ft_ci:>15} "
            f"{format_mean_std(row, 'ft_macro_f1'):>13} {ft_f1_ci:>15} "
            f"{format_mean_std(row, 'ft_macro_f1_gold'):>13} | {delta}"
        )

    print("\nPer seed (acc / macro F1 / gold-class F1, unk, McNemar vs base: FT-only-correct / base-only-correct, p):")
    for row in rows:
        name = DISPLAY_NAMES.get(row["model"], row["model"])
        for i, seed in enumerate(row["seeds"]):
            test = (
                f"  McNemar {row['ft_only_correct'][i]:>3} / {row['base_only_correct'][i]:>3}, "
                f"p = {row['mcnemar_p'][i]:.1e}"
                if "mcnemar_p" in row
                else ""
            )
            print(
                f"  {name[:22]:<22} seed {seed}: {row['seed_acc'][i]:.3f} / {row['seed_macro_f1'][i]:.3f} / "
                f"{row['seed_macro_f1_gold'][i]:.3f}, unk {row['seed_unk'][i]}{test}"
            )

    if any("vs_full_mcnemar_p" in row for row in rows):
        print(
            f"\nVs the full set ({FULL_DATASET}): Δ of seed means [paired bootstrap 95% CI] p; "
            "per seed pair full-only / ablation-only correct, McNemar p:"
        )
    for row in rows:
        if "vs_full_mcnemar_p" not in row:
            continue
        name = DISPLAY_NAMES.get(row["model"], row["model"])
        delta = (
            f"Δacc {row['vs_full_delta_acc']:+.3f} [{row['vs_full_delta_acc_ci_low']:+.3f}, "
            f"{row['vs_full_delta_acc_ci_high']:+.3f}] p {row['vs_full_delta_acc_p']:.3f}; "
            f"ΔF1 {row['vs_full_delta_macro_f1']:+.3f} [{row['vs_full_delta_macro_f1_ci_low']:+.3f}, "
            f"{row['vs_full_delta_macro_f1_ci_high']:+.3f}] p {row['vs_full_delta_macro_f1_p']:.3f}; "
            if "vs_full_delta_acc" in row
            else "(no paired Δ: seeds differ from the full set's); "
        )
        pairs = "; ".join(
            f"s{seed} {full_only}/{ablation_only} p={p_value:.2g}"
            for seed, full_only, ablation_only, p_value in zip(
                row["seeds"], row["vs_full_full_only_correct"], row["vs_full_ablation_only_correct"],
                row["vs_full_mcnemar_p"], strict=True,
            )
        )  # fmt: skip
        print(f"  {name[:22]:<22} {delta}{pairs}")

    if rows[0]["eval_set"] != "test" and any("answered_base_acc" in row for row in rows):
        print("\nItems the base model answered (not Other): base acc -> seed acc (FT-only / base-only, McNemar p):")
    for row in rows:
        if row["eval_set"] == "test" or "answered_base_acc" not in row:
            continue
        name = DISPLAY_NAMES.get(row["model"], row["model"])
        seeds = "; ".join(
            f"s{seed} {acc:.3f} ({ft_only}/{base_only}, p={p_value:.2g})"
            for seed, acc, ft_only, base_only, p_value in zip(
                row["seeds"], row["answered_seed_acc"], row["answered_ft_only_correct"],
                row["answered_base_only_correct"], row["answered_mcnemar_p"], strict=True,
            )
        )  # fmt: skip
        answered = row["n_items"] - row["base_refusals"]
        print(f"  {name[:22]:<22} {answered}/{row['n_items']} answered: base {row['answered_base_acc']:.3f} -> {seeds}")


def print_per_class(rows: list[dict], cells: list[dict]) -> None:
    all_rows = [row for row in rows if row["model"] == ALL_MODELS]
    models = list(dict.fromkeys(row["model"] for row in rows if row["model"] != ALL_MODELS))
    header = (
        f"{'label':<40} {'n':>4} | {'base P':>6} {'R':>6} {'F1':>6} {'pred':>6} | "
        f"{'FT P':>6} {'R':>6} {'F1':>6} {'pred':>6} | {'ΔF1':>6} {'improved':>8}"
    )
    print(f"Mean over {len(models)} models (fine-tuned: mean over seeds; pred: share of predictions)")
    print(header)
    print("-" * len(header))
    for row in all_rows:
        values = [
            f"{row[key]:>6.3f}" if key in row else f"{'':>6}"
            for key in ALL_MODELS_MEAN_COLUMNS[:-1]
        ]  # fmt: skip
        print(
            f"{row['label'][:40]:<40} {row['support']:>4} | {' '.join(values[:4])} | {' '.join(values[4:])} | "
            f"{row['delta_f1']:>+6.3f} {row['models_improved']:>4}/{row['n_models']}"
        )

    print("\nF1 per model, base -> fine-tuned:")
    print(f"{'label':<40} " + " ".join(f"{DISPLAY_NAMES.get(model, model)[:14]:>14}" for model in models))
    for label in [row["label"] for row in all_rows]:
        f1s = {row["model"]: row for row in rows if row["label"] == label and row["model"] != ALL_MODELS}
        print(
            f"{label[:40]:<40} "
            + " ".join(f"{f1s[model]['base_f1']:.3f}->{f1s[model]['ft_f1_mean']:.3f}" for model in models)
        )

    labels = [row["label"] for row in all_rows if row["support"] and row["label"] not in ("macro avg", "accuracy")]
    predicted = [*labels, "unk"]
    for setting in ("base", "fine-tuned"):
        shares = {
            (cell["gold"], cell["predicted"]): cell["share"]
            for cell in cells
            if cell["model"] == ALL_MODELS and cell["setting"] == setting
        }
        print(f"\nConfusion matrix, all models pooled, {setting} (rows: gold, share of the gold class's items):")
        print(f"{'':<40} " + " ".join(f"{i:>5}" for i in range(1, len(labels) + 1)) + f" {'unk':>5}")
        for i, gold in enumerate(labels, 1):
            print(f"{i:>2} {gold[:37]:<37} " + " ".join(f"{shares.get((gold, p), 0.0):>5.2f}" for p in predicted))


SUMMARY_COLUMNS = [
    "model", "eval_set", "dataset", "n_items",
    "base_acc", "base_acc_ci_low", "base_acc_ci_high",
    "base_macro_f1", "base_macro_f1_ci_low", "base_macro_f1_ci_high", "base_macro_f1_gold", "base_unk",
    "seeds", "ft_acc_mean", "ft_acc_std", "ft_acc_ci_low", "ft_acc_ci_high",
    "ft_macro_f1_mean", "ft_macro_f1_std", "ft_macro_f1_ci_low", "ft_macro_f1_ci_high",
    "ft_macro_f1_gold_mean", "ft_macro_f1_gold_std",
    "delta_acc", "delta_macro_f1",
    "seed_acc", "seed_macro_f1", "seed_macro_f1_gold", "seed_unk",
    "ft_only_correct", "base_only_correct", "mcnemar_p",
    "base_refusals", "answered_base_acc", "answered_seed_acc",
    "answered_ft_only_correct", "answered_base_only_correct", "answered_mcnemar_p",
    "vs_full_delta_acc", "vs_full_delta_acc_ci_low", "vs_full_delta_acc_ci_high", "vs_full_delta_acc_p",
    "vs_full_delta_macro_f1", "vs_full_delta_macro_f1_ci_low", "vs_full_delta_macro_f1_ci_high",
    "vs_full_delta_macro_f1_p",
    "vs_full_full_only_correct", "vs_full_ablation_only_correct", "vs_full_mcnemar_p",
]  # fmt: skip
SCORE_COLUMNS = [
    "file", "model", "eval_set", "dataset", "seed",
    "n", "acc", "macro_f1", "strict", "scan", "none", "unparseable_class", "unk",
]  # fmt: skip
PER_CLASS_COLUMNS = [
    "model", "eval_set", "dataset", "label", "support", "gold_share", "seeds", "n_models",
    "base_precision", "base_recall", "base_f1", "base_predicted_share",
    "ft_precision_mean", "ft_recall_mean", "ft_f1_mean", "ft_f1_std", "ft_predicted_share_mean",
    "delta_f1", "models_improved",
]  # fmt: skip
CONFUSION_COLUMNS = ["model", "eval_set", "dataset", "setting", "gold", "predicted", "count", "share"]


def format_value(value: object) -> object:
    """Scores to 4 significant digits (keeps small p-values), per-seed lists ";"-joined in seed order."""
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, list):
        return ";".join(str(format_value(item)) for item in value)
    return value


def write_csv(rows: list[dict], columns: list[str], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([format_value(row.get(column, "")) for column in columns])
    print(f"wrote {path}")


def aggregate_predictions(
    eval_set: str | None = None,
    dataset: str | None = None,
    models: list[str] | None = None,
    predictions_dir: str = PREDICTIONS_DIR,
    bootstrap_resamples: int = 10000,
    per_class: bool = False,
) -> None:
    """Score base and fine-tuned prediction files per model."""
    prediction_files = get_prediction_files(predictions_dir)
    tables = list(dict.fromkeys((file["eval_set"], file["dataset"]) for file in prediction_files if file["dataset"]))
    if per_class:
        selected = [
            (table_eval_set, table_dataset)
            for table_eval_set, table_dataset in tables
            if (not eval_set or table_eval_set == eval_set) and (not dataset or table_dataset == dataset)
        ]
        per_class_rows, confusion_rows = aggregate_per_class_tables(selected, models or MODELS, predictions_dir, True)
        if not per_class_rows:
            print("no prediction files found")
            raise typer.Exit(1)
        if eval_set or dataset or models:
            return
        print()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        write_csv(per_class_rows, PER_CLASS_COLUMNS, f"{RESULTS_DIR}/per_class.csv")
        write_csv(confusion_rows, CONFUSION_COLUMNS, f"{RESULTS_DIR}/confusion.csv")
        return

    rows = []
    for table_eval_set, table_dataset in tables:
        if (eval_set and table_eval_set != eval_set) or (dataset and table_dataset != dataset):
            continue
        table = []
        for model in models or MODELS:
            row = aggregate_model(predictions_dir, model, table_eval_set, table_dataset, bootstrap_resamples)
            if row is not None:
                table.append(row)
        if table:
            print(f"\n=== eval set {table_eval_set}, fine-tuning dataset {table_dataset}, {table[0]['n_items']} items")
            print_summary(table)
            rows += table
    if not rows:
        print("no prediction files found")
        raise typer.Exit(1)
    if eval_set or dataset or models:
        return

    print()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    scores = [{**score(file["path"]), **file} | {"file": Path(file["path"]).name} for file in prediction_files]
    write_csv(scores, SCORE_COLUMNS, f"{RESULTS_DIR}/scores.csv")
    write_csv(rows, SUMMARY_COLUMNS, f"{RESULTS_DIR}/summary.csv")
    per_class_rows, confusion_rows = aggregate_per_class_tables(tables, MODELS, predictions_dir, False)
    write_csv(per_class_rows, PER_CLASS_COLUMNS, f"{RESULTS_DIR}/per_class.csv")
    write_csv(confusion_rows, CONFUSION_COLUMNS, f"{RESULTS_DIR}/confusion.csv")


if __name__ == "__main__":
    typer.run(aggregate_predictions)
