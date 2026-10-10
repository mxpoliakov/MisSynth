import glob
import json
import os
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import typer
from rouge_score import tokenizers

from missci.util.fileutil import read_jsonl


def get_synthetic_classes(row: dict[str, any]) -> list[str]:
    """Extractor for fallacy classes from a MisSynth dataset row."""
    classes: list[str] = []
    for synthetic_fallacy in row.get("synthetic_fallacies", []):
        fallacy_class: str = synthetic_fallacy.get("class", "")

        # Unify class names
        if fallacy_class == "Fallacy of Composition":
            fallacy_class = "Fallacy of Division/Composition"
        elif fallacy_class == "False Dilemma":
            fallacy_class = "False Dilemma / Affirming the Disjunct"

        if fallacy_class:
            classes.append(fallacy_class)
    return classes


def get_missci_classes(row: dict[str, any]) -> list[str]:
    """Extractor for fallacy classes from a MISSCI dataset row."""
    classes: list[str] = []
    for fallacy_dict in row.get("argument", {}).get("fallacies", []):
        for fallacy in fallacy_dict.get("interchangeable_fallacies", []):
            if fallacy.get("class"):
                classes.append(fallacy["class"])
    return classes


def count_fallacy_classes(
    dataset_filename: str,
    class_extractor_func: Callable[[dict[str, any]], list[str]],
) -> Counter[str]:
    """
    Reads a dataset and counts fallacy classes using a provided extractor function.
    """
    dataset: Iterable[dict[str, any]] = read_jsonl(dataset_filename)
    all_classes: list[str] = []
    for row in dataset:
        all_classes.extend(class_extractor_func(row))
    return Counter(all_classes)


# ROUGE-1 recall from token counts, so that every entity can be scored against all 30 excerpts.
tokenizer = tokenizers.DefaultTokenizer(use_stemmer=True)

# Comparison generations; skipped if a folder is missing.
COMPARISON_FOLDERS = {
    "o4-mini no-retrieval": "o4-mini-single-class-synthetic-fallacy-context-no-retrieval-15",
    "o4-mini random-source": "o4-mini-single-class-synthetic-fallacy-context-random-source-15",
}
RANDOM_SOURCE_MAP = "output/o4-mini-single-class-synthetic-fallacy-context-random-source-15/random_source_map.json"


def rouge1_recall(entity_counts: Counter[str], excerpt_counts: Counter[str]) -> float:
    overlap = sum(min(count, excerpt_counts[token]) for token, count in entity_counts.items())
    return overlap / sum(entity_counts.values())


def score_entities(pairs: list[tuple[str, str]], excerpts: dict[str, Counter[str]]) -> dict[str, np.ndarray]:
    """Per-item ROUGE-1 recall against the own excerpt and against every other argument's excerpt."""
    argument_ids = sorted(excerpts)
    rows = []
    for argument_id, text in pairs:
        entity_counts = Counter(tokenizer.tokenize(text))
        if not entity_counts:
            continue
        recalls = np.array([rouge1_recall(entity_counts, excerpts[a]) for a in argument_ids])
        own_index = argument_ids.index(argument_id)
        others = np.delete(recalls, own_index)
        top = argument_ids[int(np.argmax(recalls))]
        rows.append((argument_id, recalls[own_index], others.mean(), float(recalls[own_index] > others.max()), top))
    return {
        "argument": np.array([row[0] for row in rows]),
        "own": np.array([row[1] for row in rows]),
        "other": np.array([row[2] for row in rows]),
        "excess": np.array([row[1] - row[2] for row in rows]),
        "attributed": np.array([row[3] for row in rows]),
        "top": np.array([row[4] for row in rows]),
    }


def get_bootstrap_ci(
    scores: dict[str, np.ndarray], metric: str, n_resamples: int, baseline: dict[str, np.ndarray] | None = None
) -> tuple[float, float]:
    """95% CI of an item mean, or of its paired difference to a baseline, resampling dev arguments."""
    rng = np.random.default_rng(0)
    argument_ids = sorted(set(scores["argument"]))
    by_argument = {a: scores[metric][scores["argument"] == a] for a in argument_ids}
    baseline_by_argument = (
        {a: baseline[metric][baseline["argument"] == a] for a in argument_ids} if baseline is not None else None
    )
    estimates = []
    for _ in range(n_resamples):
        sample = rng.choice(argument_ids, len(argument_ids))
        estimate = np.concatenate([by_argument[a] for a in sample]).mean()
        if baseline_by_argument is not None:
            estimate -= np.concatenate([baseline_by_argument[a] for a in sample]).mean()
        estimates.append(estimate)
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def get_raw_pairs(folder: str, field: str) -> list[tuple[str, str]]:
    """(argument id, text) for every synthetic fallacy in a raw output folder."""
    pairs = []
    for path in sorted(glob.glob(f"output/{folder}/raw/*.json")):
        with open(path) as f:
            items = json.load(f)
        pairs += [(Path(path).stem, item[field]) for item in items if item.get(field)]
    return pairs


def get_release_pairs(dataset: list[dict[str, any]], entity_key: str, sub_entity_key: str) -> list[tuple[str, str]]:
    return [
        (row["id"], item[sub_entity_key])
        for row in dataset
        for item in row.get(entity_key) or []
        if item.get(sub_entity_key)
    ]


def get_missci_pairs(
    missci_dataset: Iterable[dict[str, any]], text_extractor_func: Callable[[dict[str, any]], list[str]]
) -> list[tuple[str, str]]:
    return [(row["id"], text) for row in missci_dataset for text in text_extractor_func(row) if text]


def extract_missci_fallacy(row: dict[str, any]) -> list[str]:
    return [
        f["premise"]
        for fd in row.get("argument", {}).get("fallacies", [])
        for f in fd.get("interchangeable_fallacies", [])
    ]


def extract_missci_context(row: dict[str, any]) -> list[str]:
    return [fd["fallacy_context"] for fd in row.get("argument", {}).get("fallacies", []) if fd.get("fallacy_context")]


def extract_missci_claim(row: dict[str, any]) -> list[str]:
    return [row.get("argument", {}).get("claim")]


def extract_missci_premise(row: dict[str, any]) -> list[str]:
    return [row.get("argument", {}).get("accurate_premise_p0", {}).get("premise")]


def analyze_synthetic_dataset(
    model: str = "o4-mini",
    compare_models: list[str] = ["gpt-5"],  # noqa: B006
    bootstrap_resamples: int = 5000,
    csv_file: str | None = "results/source_attribution.csv",
) -> None:
    print("--- Fallacy Class Counts ---")
    synth_dataset_file: str = f"dataset/MisSynth.{model}.jsonl"
    missci_dev_file: str = "missci/dataset/dev.missci.jsonl"
    missci_test_file: str = "missci/dataset/test.missci.jsonl"

    c1: Counter[str] = count_fallacy_classes(synth_dataset_file, get_synthetic_classes)
    c2: Counter[str] = count_fallacy_classes(missci_dev_file, get_missci_classes)
    c3: Counter[str] = count_fallacy_classes(missci_test_file, get_missci_classes)

    df: pd.DataFrame = pd.DataFrame([c1, c2, c3]).T.fillna(0).astype(int)
    df.columns = [f"MisSynth {model}", "MISSCI (dev)", "MISSCI (test)"]
    df_copy: pd.DataFrame = df.copy()
    df.loc["Overall"] = df.sum()
    print(df)

    print("\n--- Normalized Fallacy Class Counts (%) ---")
    df_normalized: pd.DataFrame = df_copy.div(df_copy.sum(axis=0), axis=1)
    print(df_normalized.round(4) * 100)

    print("\n--- ROUGE-1 recall against the article excerpt ---")
    synthetic_dataset: list[dict[str, any]] = list(read_jsonl(synth_dataset_file))
    missci_dev_dataset: list[dict[str, any]] = list(read_jsonl(missci_dev_file))
    excerpts = {row["id"]: Counter(tokenizer.tokenize(row["rag_article_excerpt"])) for row in synthetic_dataset}

    entities: dict[str, list[tuple[str, str]]] = {}
    for name, extractor in [
        ("fallacy", extract_missci_fallacy),
        ("context", extract_missci_context),
        ("claim", extract_missci_claim),
        ("premise", extract_missci_premise),
    ]:
        entities[f"MISSCI (dev) {name}"] = get_missci_pairs(missci_dev_dataset, extractor)
    for name, entity_key, sub_entity_key in [
        ("fallacy", "synthetic_fallacies", "fallacy"),
        ("context", "synthetic_fallacies", "context"),
        ("claim", "synthetic_claims_and_premises", "claim"),
        ("premise", "synthetic_claims_and_premises", "premise"),
    ]:
        entities[f"MisSynth {model} {name}"] = get_release_pairs(synthetic_dataset, entity_key, sub_entity_key)
    for label, folder in COMPARISON_FOLDERS.items():
        if not os.path.isdir(f"output/{folder}/raw"):
            print(f"skip {label}: output/{folder} not found")
            continue
        for field in ("fallacy", "context"):
            entities[f"{label} {field}"] = get_raw_pairs(folder, field)
    for compare_model in compare_models:
        compare_dataset = list(read_jsonl(f"dataset/MisSynth.{compare_model}.jsonl"))
        for field in ("fallacy", "context"):
            entities[f"MisSynth {compare_model} {field}"] = get_release_pairs(
                compare_dataset, "synthetic_fallacies", field
            )

    scores = {name: score_entities(pairs, excerpts) for name, pairs in entities.items()}
    print(
        "own = recall against the own excerpt; other = mean against the other excerpts; excess = own - other;\n"
        f"attributed = own excerpt ranks first of {len(excerpts)} (chance {1 / len(excerpts):.3f}); "
        "95% CIs resample the dev arguments"
    )
    rows = []
    for name, entity_scores in scores.items():
        row = {"entity": name, "n": len(entity_scores["own"])}
        for metric in ("own", "other", "excess", "attributed"):
            row[metric] = entity_scores[metric].mean()
        for metric in ("excess", "attributed"):
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = get_bootstrap_ci(
                entity_scores, metric, bootstrap_resamples
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    print(table.round(3).to_string(index=False))

    if "o4-mini random-source context" in scores and os.path.exists(RANDOM_SOURCE_MAP):
        with open(RANDOM_SOURCE_MAP) as f:
            assigned = {a: entry["source_id"] for a, entry in json.load(f)["map"].items()}
        random_source = scores["o4-mini random-source context"]
        to_assigned = np.mean([assigned[a] == top for a, top in zip(random_source["argument"], random_source["top"])])
        print(f"\no4-mini random-source context: the assigned article's excerpt ranks first for {to_assigned:.3f}")

    print("\n--- Paired differences (same arguments resampled) ---")
    contrasts = [
        (f"MisSynth {model} context", "o4-mini no-retrieval context"),
        (f"MisSynth {model} context", "o4-mini random-source context"),
        (f"MisSynth {model} context", "MISSCI (dev) context"),
        (f"MisSynth {model} fallacy", "MISSCI (dev) fallacy"),
    ] + [(f"MisSynth {compare_model} context", f"MisSynth {model} context") for compare_model in compare_models]
    for name, baseline_name in contrasts:
        if name not in scores or baseline_name not in scores or name == baseline_name:
            continue
        for metric in ("excess", "attributed"):
            difference = scores[name][metric].mean() - scores[baseline_name][metric].mean()
            low, high = get_bootstrap_ci(scores[name], metric, bootstrap_resamples, scores[baseline_name])
            print(f"{name} - {baseline_name}, {metric}: {difference:+.3f} [{low:+.3f}, {high:+.3f}]")

    if csv_file:
        os.makedirs(os.path.dirname(csv_file) or ".", exist_ok=True)
        table.to_csv(csv_file, index=False)
        print(f"\nwrote {csv_file}")


if __name__ == "__main__":
    typer.run(analyze_synthetic_dataset)
