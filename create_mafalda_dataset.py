"""Convert MAFALDA gold-standard dataset into MISSCI-compatible evaluation format.

MAFALDA (NAACL 2024) uses span-level fallacy annotations on text paragraphs
with 23 Level-2 fallacy classes.  This script maps MAFALDA fallacy classes to
the MISSCI 9-class taxonomy, filters out unmappable classes, and produces an
evaluation JSONL in the same record structure as missci/dataset/*.missci.jsonl

Mapping rationale
-----------------
Only 7 of 23 MAFALDA classes have a reasonable semantic overlap with the
MISSCI taxonomy (which targets scientific-communication fallacies) and map to
6 MISSCI classes. The 16 remaining classes (mostly emotion- and credibility-based)
have no MISSCI counterpart and are skipped, as are the "nothing" and "to clean" pseudo-labels.

Unmapped MAFALDA classes (16):
    appeal to positive emotion, appeal to anger, appeal to fear, appeal to
    pity, appeal to ridicule, appeal to worse problems, circular reasoning,
    slippery slope, straw man, ad hominem, ad populum, appeal to (false)
    authority, appeal to nature, appeal to tradition, guilt by association,
    tu quoque

MISSCI classes with no MAFALDA equivalent (3):
    Impossible Expectations, Biased Sample Fallacy, Fallacy of Exclusion
"""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import typer

MAFALDA_TO_MISSCI: dict[str, str] = {
    "equivocation": "Ambiguity",
    "causal oversimplification": "Causal Oversimplification",
    "false causality": "Causal Oversimplification",
    "false dilemma": "False Dilemma / Affirming the Disjunct",
    "hasty generalization": "Hasty Generalization",
    "false analogy": "False Equivalence",
    "fallacy of division": "Fallacy of Division/Composition",
}

MISSCI_CLASSES_COVERED = sorted(set(MAFALDA_TO_MISSCI.values()))

MAFALDA_DATASET_URL = "https://raw.githubusercontent.com/ChadiHelwe/MAFALDA/main/datasets/gold_standard_dataset.jsonl"

DEFAULT_OUTPUT_PATH = "dataset/mafalda.test.jsonl"


def fetch_mafalda_dataset() -> list[dict]:
    """Download the MAFALDA gold-standard JSONL into memory (not saved to disk)."""
    print("Downloading MAFALDA dataset …")
    with urllib.request.urlopen(MAFALDA_DATASET_URL) as response:
        raw = response.read()

    records: list[dict] = []
    for line in io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8"):
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _extract_span_text(text: str, start: int, end: int) -> str:
    return text[start:end].strip()


def _write_jsonl(path: str, records: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def convert_mafalda_to_missci(records: list[dict]) -> list[dict]:
    """Convert MAFALDA records into the MISSCI record structure.

    Each MAFALDA record has a ``text`` and ``labels`` list of
    ``[start, end, class_name]`` triples.  A single record can contain
    multiple fallacy spans; we group all mappable spans from the same
    MAFALDA record into a single MISSCI-style record so that the
    ``argument.fallacies`` list faithfully mirrors the original structure.

    Because MAFALDA texts are *not* structured as claim + premises, we
    adopt the following adaptation:

      - **claim**: the full paragraph text
      - **accurate_premise_p0**: the full paragraph text
      - **fallacy_context**: the full paragraph text (Premise 2 / context)
      - **fallacious_premise**: the annotated fallacious span text

    This lets the model see the full text and focus on the flagged span.
    """
    output: list[dict] = []
    skipped_classes: dict[str, int] = {}

    for idx, record in enumerate(records):
        text: str = record["text"]
        labels: list = record.get("labels", [])

        if not labels:
            continue

        fallacies: list[dict] = []
        for label in labels:
            if len(label) < 3:
                continue
            start, end, mafalda_class = label[0], label[1], label[2].lower().strip()

            # Skip the "nothing" and "to clean" pseudo-labels
            if mafalda_class in ("nothing", "to clean"):
                continue

            missci_class = MAFALDA_TO_MISSCI.get(mafalda_class)
            if missci_class is None:
                skipped_classes[mafalda_class] = skipped_classes.get(mafalda_class, 0) + 1
                continue

            fallacious_span = _extract_span_text(text, start, end)
            if not fallacious_span:
                continue

            fallacy_id = f"mafalda-{idx}:{start}-{end}"
            fallacies.append(
                {
                    "fallacy_context": text.strip(),
                    "id": fallacy_id,
                    "interchangeable_fallacies": [
                        {
                            "premise": fallacious_span,
                            "class": missci_class,
                            "id": f"{fallacy_id}:1",
                            "source_class": mafalda_class,
                        }
                    ],
                }
            )

        if not fallacies:
            continue

        output.append(
            {
                "id": f"mafalda-{idx}",
                "argument": {
                    "claim": text.strip(),
                    "accurate_premise_p0": {
                        "premise": text.strip(),
                    },
                    "fallacies": fallacies,
                },
            }
        )

    if skipped_classes:
        print("\nSkipped MAFALDA classes (not in MISSCI taxonomy):")
        for cls, count in sorted(skipped_classes.items(), key=lambda x: -x[1]):
            print(f"  {cls}: {count}")

    return output


def create_mafalda_dataset(
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> None:
    """Download MAFALDA, map fallacy classes to MISSCI taxonomy, write eval JSONL."""

    records = fetch_mafalda_dataset()
    print(f"Loaded {len(records)} MAFALDA records")

    eval_entries = convert_mafalda_to_missci(records)

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
        print(f"MISSCI classes with no MAFALDA samples: {sorted(missing)}")


if __name__ == "__main__":
    typer.run(create_mafalda_dataset)
