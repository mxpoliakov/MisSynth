"""Convert the Logic (causalNLP/logical-fallacy) dataset into MISSCI-compatible evaluation format.

The Logic dataset (EMNLP 2022 Findings) contains sentence-level fallacy
annotations across 13 fallacy classes.  This script maps Logic fallacy classes
to the MISSCI 9-class taxonomy, filters out unmappable classes, and produces an
evaluation JSONL in the same record structure as missci/dataset/*.missci.jsonl

Data source
-----------
CSV files from https://github.com/causalNLP/logical-fallacy/tree/main/data
    - edu_train.csv, edu_dev.csv, edu_test.csv  (label column: ``updated_label``)
    - climate_train.csv, climate_dev.csv, climate_test.csv  (label column: ``logical_fallacies``)
Each row has a ``source_article`` (text) column and a fallacy label column.

Mapping rationale
-----------------
Only 4 of 13 Logic classes have a one-to-one counterpart in the MISSCI
taxonomy (which targets scientific-communication fallacies).  The 9 remaining
classes have no MISSCI counterpart and are skipped.

Mapped Logic → MISSCI classes (4):
    equivocation            → Ambiguity
    false causality         → Causal Oversimplification
    false dilemma           → False Dilemma / Affirming the Disjunct
    faulty generalization   → Hasty Generalization

Unmapped Logic classes (9):
    ad hominem, ad populum, appeal to emotion, fallacy of credibility,
    fallacy of extension, fallacy of relevance, intentional,
    circular reasoning, fallacy of logic

MISSCI classes with no Logic equivalent (5):
    Biased Sample Fallacy, Fallacy of Division/Composition,
    Fallacy of Exclusion, Impossible Expectations, False Equivalence
"""

from __future__ import annotations

import csv
import io
import json
import urllib.request
from pathlib import Path

import typer

LOGIC_TO_MISSCI: dict[str, str] = {
    "equivocation": "Ambiguity",
    "false causality": "Causal Oversimplification",
    "false dilemma": "False Dilemma / Affirming the Disjunct",
    "faulty generalization": "Hasty Generalization",
}

MISSCI_CLASSES_COVERED = sorted(set(LOGIC_TO_MISSCI.values()))

# Raw CSV URLs from the causalNLP/logical-fallacy repository
_BASE_URL = "https://raw.githubusercontent.com/causalNLP/logical-fallacy/main/data"

# Each entry: (url, label_column)
LOGIC_DATASET_SOURCES: list[tuple[str, str, str]] = [
    # Edu splits (label column: updated_label)
    ("edu-train", f"{_BASE_URL}/edu_train.csv", "updated_label"),
    ("edu-dev", f"{_BASE_URL}/edu_dev.csv", "updated_label"),
    ("edu-test", f"{_BASE_URL}/edu_test.csv", "updated_label"),
    # Climate splits (label column: logical_fallacies)
    ("climate-train", f"{_BASE_URL}/climate_train.csv", "logical_fallacies"),
    ("climate-dev", f"{_BASE_URL}/climate_dev.csv", "logical_fallacies"),
    ("climate-test", f"{_BASE_URL}/climate_test.csv", "logical_fallacies"),
]

DEFAULT_OUTPUT_PATH = "dataset/logic.test.jsonl"


def fetch_logic_dataset() -> list[dict]:
    """Download all Logic (edu + climate) CSV splits into memory and return as records.

    Each record is a dict with at least ``source_article`` and ``updated_label``.
    """
    records: list[dict] = []
    for split, url, label_col in LOGIC_DATASET_SOURCES:
        print(f"Downloading Logic {split} split …")
        with urllib.request.urlopen(url) as response:
            raw = response.read()
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
        for row in reader:
            article = row.get("source_article", "").strip()
            label = row.get(label_col, "").strip()
            if article and label:
                records.append(
                    {
                        "source_article": article,
                        "updated_label": label,
                        "split": split,
                    }
                )
    return records


def _write_jsonl(path: str, records: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def convert_logic_to_missci(records: list[dict]) -> list[dict]:
    """Convert Logic records into the MISSCI record structure.

    Each Logic record is a single (text, label) pair.  We adapt it as follows:

      - **claim**: the full source article text
      - **accurate_premise_p0**: the full source article text
      - **fallacy_context**: the full source article text
      - **fallacious_premise**: the full source article text (entire sentence
        is the fallacious content)

    This matches the adaptation strategy used in create_mafalda_dataset.py.
    """
    output: list[dict] = []
    skipped_classes: dict[str, int] = {}

    for idx, record in enumerate(records):
        text: str = record["source_article"]
        logic_class: str = record["updated_label"].lower().strip()

        missci_class = LOGIC_TO_MISSCI.get(logic_class)
        if missci_class is None:
            skipped_classes[logic_class] = skipped_classes.get(logic_class, 0) + 1
            continue

        record_id = f"logic-{idx}"
        fallacy_id = f"logic-{idx}:0-{len(text)}"

        output.append(
            {
                "id": record_id,
                "argument": {
                    "claim": text,
                    "accurate_premise_p0": {
                        "premise": "(see full text in claim)",
                    },
                    "fallacies": [
                        {
                            "fallacy_context": text,
                            "id": fallacy_id,
                            "interchangeable_fallacies": [
                                {
                                    "premise": "(see full text in claim)",
                                    "class": missci_class,
                                    "id": f"{fallacy_id}:1",
                                }
                            ],
                        }
                    ],
                },
            }
        )

    if skipped_classes:
        print("\nSkipped Logic classes (not in MISSCI taxonomy):")
        for cls, count in sorted(skipped_classes.items(), key=lambda x: -x[1]):
            print(f"  {cls}: {count}")

    return output


def create_logic_dataset(
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> None:
    """Download Logic dataset, map fallacy classes to MISSCI taxonomy, write eval JSONL."""

    records = fetch_logic_dataset()
    print(f"Loaded {len(records)} Logic records")

    eval_entries = convert_logic_to_missci(records)

    _write_jsonl(output_path, eval_entries)
    print(f"\nWrote {len(eval_entries)} MISSCI-format records to {output_path}")

    class_counts: dict[str, int] = {}
    total_fallacies = 0
    for entry in eval_entries:
        for fallacy in entry["argument"]["fallacies"]:
            for interchangeable in fallacy["interchangeable_fallacies"]:
                cls = interchangeable["class"]
                class_counts[cls] = class_counts.get(cls, 0) + 1
                total_fallacies += 1

    print(f"\nTotal fallacy annotations: {total_fallacies}")
    print("Mapped class distribution:")
    for cls in sorted(class_counts.keys()):
        print(f"  {cls}: {class_counts[cls]}")

    print(f"\nMISSCI classes present:  {len(class_counts)}/{len(MISSCI_CLASSES_COVERED)}")
    missing = set(MISSCI_CLASSES_COVERED) - set(class_counts.keys())
    if missing:
        print(f"MISSCI classes with no Logic samples: {sorted(missing)}")


if __name__ == "__main__":
    typer.run(create_logic_dataset)
