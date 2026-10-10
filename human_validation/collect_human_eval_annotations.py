import csv
import itertools
import math
import re

import typer
from sklearn.metrics import cohen_kappa_score

from missci.util.fallacy_util import get_valid_fallacy_names
from missci.util.fileutil import read_jsonl
from missci.util.fileutil import write_jsonl

QUESTION_FIELDS = {
    "1. Fallacious?": "fallacious",
    "2. Grounded?": "grounded",
    "3. Class": "classes",
    "Comment": "comment",
}
REQUIRED_FIELDS = ["fallacious", "grounded", "classes"]
GROUNDED_CHOICES = {
    "Supported": "supported",
    "Not in the excerpt": "not_in_excerpt",
    "Contradicts the excerpt": "contradicts",
}
NONE_OF_THESE = "None of these"
TITLE_PATTERN = re.compile(r"^\[(?P<id>[^\]]+)\] (?P<question>.+)$")
# page 2 of the form; "No" submits without any items
CONFIRMATION_TITLE = "I have read the guidelines, will do the annotation myself and want to continue"
# page 1 of the form: the annotator's initials
ANNOTATOR_TITLE = "Annotator"


def read_responses(responses_csv: str) -> list[tuple[str, dict]]:
    """Annotator and answers of each confirmed response in a form-response CSV."""
    responses = []
    with open(responses_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        titles = next(reader)
        for row_number, values in enumerate(reader, start=2):
            row = {}
            for title, value in zip(titles, values, strict=False):
                assert not (value.strip() and row.get(title, "").strip()), f"{responses_csv}: two answers for {title}"
                if value.strip() or title not in row:
                    row[title] = value
            if row[CONFIRMATION_TITLE] != "Yes":
                print(f"{responses_csv}: row {row_number} did not confirm the guidelines page, response skipped")
                continue
            answers = {}
            for title, value in row.items():
                match = TITLE_PATTERN.match(title or "")
                if match is None or match["question"] not in QUESTION_FIELDS:
                    continue
                answers.setdefault(match["id"], {})[QUESTION_FIELDS[match["question"]]] = value.strip()
            responses.append((row[ANNOTATOR_TITLE].strip(), answers))
    return responses


def parse_answers(item_answers: dict, where: str) -> dict:
    missing = [field for field in REQUIRED_FIELDS if not item_answers.get(field)]
    assert not missing, f"{where}: no answer for {missing}"
    assert item_answers["fallacious"] in {"Yes", "No"}, f"{where}: {item_answers['fallacious']}"
    assert item_answers["grounded"] in GROUNDED_CHOICES, f"{where}: {item_answers['grounded']}"
    # Google Forms joins checkbox answers with ", "; no class name contains a comma
    classes = item_answers["classes"].split(", ")
    valid_classes = [*get_valid_fallacy_names(include_other=False), NONE_OF_THESE]
    assert all(c in valid_classes for c in classes), f"{where}: {classes}"
    return {
        "fallacious": item_answers["fallacious"] == "Yes",
        "grounded": GROUNDED_CHOICES[item_answers["grounded"]],
        "classes": classes,
        "comment": item_answers.get("comment", ""),
    }


def find_inconsistency(record: dict) -> str | None:
    """Code for an answer that breaks the class rule (at most two classes, "None of these" alone)."""
    classes = record["classes"]
    if NONE_OF_THESE in classes and len(classes) > 1:
        return "NONE_WITH_CLASS"
    if len(classes) > 2:
        return "TOO_MANY_CLASSES"
    return None


def format_rate(k: int, n: int, z: float = 1.96) -> str:
    """k/n with a Wilson 95% interval."""
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{p:.2f} [{center - half_width:.2f}, {center + half_width:.2f}] ({k}/{n})"


def get_class_labels(first: list[str], second: list[str]) -> tuple[str, str]:
    """One class per side for kappa: a class both ticked if there is one, otherwise each side's first tick."""
    shared = [c for c in first if c in second]
    return (shared[0], shared[0]) if shared else (first[0], second[0])


def print_agreement(name: str, first: list, second: list) -> None:
    agreement = format_rate(sum(a == b for a, b in zip(first, second, strict=True)), len(first))
    print(f"  {name}: agreement {agreement}, Cohen's kappa {cohen_kappa_score(first, second):.2f}")


def collect_human_eval_annotations(responses_csv: list[str], model_name: str = "o4-mini") -> None:
    """Score the form responses against the generator's class and, with several annotators, pairwise.

    Answers that break the guidelines are kept in the records but not scored.
    """
    sample = {item["id"]: item for item in read_jsonl(f"human_validation/MisSynth.{model_name}.human_eval.jsonl")}
    responses = [response for path in responses_csv for response in read_responses(path)]
    assert responses, "no confirmed responses"
    annotators = [annotator for annotator, _ in responses]
    assert all(annotators), "every response needs the annotator's initials"
    assert len(set(annotators)) == len(annotators), f"annotators must have unique initials: {annotators}"

    annotations = []
    problems = []
    for annotator, answers in responses:
        unknown = set(answers) - set(sample)
        assert not unknown, f"annotator {annotator}: answers for items not in the sample: {sorted(unknown)}"
        for item_id in sample:
            record = {"id": item_id, "annotator": annotator}
            try:
                record |= parse_answers(answers.get(item_id, {}), f"annotator {annotator}, {item_id}")
            except AssertionError as error:
                problems.append(str(error))
                continue
            record["inconsistency"] = find_inconsistency(record)
            annotations.append(record)
    # report every problem at once
    assert not problems, "answers to fix in the form:\n" + "\n".join(problems)

    output_path = f"human_validation/MisSynth.{model_name}.human_eval.annotations.jsonl"
    write_jsonl(output_path, annotations)
    print(f"{len(annotations)} annotations ({len(responses)} annotator(s) x {len(sample)} items) -> {output_path}")
    for record in annotations:
        if record["inconsistency"]:
            print(
                f"annotator {record['annotator']}, {record['id']}: "
                f"{record['inconsistency']} {record['classes']}, left out"
            )

    by_annotator = {
        annotator: {
            record["id"]: record
            for record in annotations
            if record["annotator"] == annotator and not record["inconsistency"]
        }
        for annotator in annotators
    }
    for annotator, records in by_annotator.items():
        n = len(records)
        generator_classes = [[sample[item_id]["training_class"]] for item_id in records]
        classes = [record["classes"] for record in records.values()]
        labels = [get_class_labels(c, g) for c, g in zip(classes, generator_classes, strict=True)]
        print(f"annotator {annotator}:")
        print(f"  fallacious: {format_rate(sum(record['fallacious'] for record in records.values()), n)}")
        for choice, value in GROUNDED_CHOICES.items():
            count = sum(record["grounded"] == value for record in records.values())
            print(f"  grounded, {choice}: {format_rate(count, n)}")
        print_agreement("class vs generator (generator class ticked)", *map(list, zip(*labels, strict=True)))
        only_ticked = sum(c == g for c, g in zip(classes, generator_classes, strict=True))
        print(f"  only the generator class ticked: {format_rate(only_ticked, n)}")
    for first, second in itertools.combinations(by_annotator, 2):
        shared = [item_id for item_id in by_annotator[first] if item_id in by_annotator[second]]
        print(f"annotator {first} vs annotator {second} ({len(shared)} items):")
        for field in ["fallacious", "grounded"]:
            print_agreement(
                field, [by_annotator[first][i][field] for i in shared], [by_annotator[second][i][field] for i in shared]
            )
        labels = [
            get_class_labels(by_annotator[first][i]["classes"], by_annotator[second][i]["classes"]) for i in shared
        ]
        print_agreement("class", *map(list, zip(*labels, strict=True)))


if __name__ == "__main__":
    typer.run(collect_human_eval_annotations)
