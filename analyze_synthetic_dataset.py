from collections import Counter
from collections.abc import Iterable
from typing import Callable

import numpy as np
import pandas as pd
import typer
from rouge_score import rouge_scorer

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


scorer: rouge_scorer.RougeScorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)


def build_excerpt_map(
    synthetic_dataset_list: list[dict[str, any]],
) -> dict[str, str]:
    """Creates a dictionary mapping row ID to its RAG article excerpt from a loaded list."""
    return {
        row["id"]: row["rag_article_excerpt"]
        for row in synthetic_dataset_list
        if "id" in row and "rag_article_excerpt" in row
    }


def calculate_missci_rouge(
    missci_dataset: Iterable[dict[str, any]],
    excerpt_map: dict[str, str],
    text_extractor_func: Callable[[dict[str, any]], list[str]],
) -> float:
    """
    Calculates mean ROUGE-1 recall for texts from a MISSCI dataset
    against corresponding RAG excerpts.
    """
    scores: list[float] = []
    for row in missci_dataset:
        rag_article_excerpt: str | None = excerpt_map.get(row.get("id"))

        if not rag_article_excerpt:
            continue

        target_texts: list[str] = text_extractor_func(row)
        for text in target_texts:
            if text:
                score_dict = scorer.score(text, rag_article_excerpt)
                scores.append(score_dict["rouge1"].recall)

    return np.mean(scores) if scores else 0.0


def calculate_synthetic_rouge(
    synthetic_dataset: Iterable[dict[str, any]],
    entity_key: str,
    sub_entity_key: str,
) -> float:
    """
    Calculates mean ROUGE-1 recall for synthetic texts
    against their own RAG excerpt in the same row.
    """
    scores: list[float] = []
    for row in synthetic_dataset:
        rag_article_excerpt: str | None = row.get("rag_article_excerpt")
        if not rag_article_excerpt:
            continue

        for item in row.get(entity_key, []):
            text_to_score: str | None = item.get(sub_entity_key)
            if text_to_score:
                score_dict = scorer.score(text_to_score, rag_article_excerpt)
                scores.append(score_dict["rouge1"].recall)

    return np.mean(scores) if scores else 0.0


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


def analyze_synthetic_dataset(model: str = "o4-mini") -> None:
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

    print("\n--- ROUGE-1 Recall Scores ---")

    synthetic_dataset: list[dict[str, any]] = list(read_jsonl(synth_dataset_file))
    missci_dev_dataset: list[dict[str, any]] = list(read_jsonl(missci_dev_file))

    excerpt_map: dict[str, str] = build_excerpt_map(synthetic_dataset)

    print(
        f"ROUGE-1 Recall, MisSynth ({model}) fallacy:",
        calculate_synthetic_rouge(synthetic_dataset, "synthetic_fallacies", "fallacy"),
    )
    print(
        f"ROUGE-1 Recall, MisSynth ({model}) context:",
        calculate_synthetic_rouge(synthetic_dataset, "synthetic_fallacies", "context"),
    )
    print(
        f"ROUGE-1 Recall, MisSynth ({model}) claim:",
        calculate_synthetic_rouge(synthetic_dataset, "synthetic_claims_and_premises", "claim"),
    )
    print(
        f"ROUGE-1 Recall, MisSynth ({model}) premise:",
        calculate_synthetic_rouge(synthetic_dataset, "synthetic_claims_and_premises", "premise"),
    )

    print(
        "ROUGE-1 Recall, MISSCI (dev) fallacy:",
        calculate_missci_rouge(missci_dev_dataset, excerpt_map, extract_missci_fallacy),
    )
    print(
        "ROUGE-1 Recall, MISSCI (dev) context:",
        calculate_missci_rouge(missci_dev_dataset, excerpt_map, extract_missci_context),
    )
    print(
        "ROUGE-1 Recall, MISSCI (dev) claim:",
        calculate_missci_rouge(missci_dev_dataset, excerpt_map, extract_missci_claim),
    )
    print(
        "ROUGE-1 Recall, MISSCI (dev) premise:",
        calculate_missci_rouge(missci_dev_dataset, excerpt_map, extract_missci_premise),
    )


if __name__ == "__main__":
    typer.run(analyze_synthetic_dataset)
