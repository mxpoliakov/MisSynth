"""Score fallacy-classification prediction files: accuracy, macro F1 and answer parsing.

An answer is parsed from a ``Fallacy: <class>`` line, else from the first class name in it (a unique
whole-word prefix of a class name counts), else as ``Other``. Predicted classes outside the gold labels
become ``unk``, which is included in the macro average, as in MISSCI.
"""

import glob as globlib
import random
import re
from collections.abc import Sequence
from math import comb
from pathlib import Path

import typer
from sklearn.metrics import classification_report

from missci.output_parser.llm_output_parser_fallacy import get_all_fallacy_names_from_prediction
from missci.output_parser.llm_output_parser_fallacy import get_transform_other_dict
from missci.util.fallacy_util import get_valid_fallacy_names
from missci.util.fallacy_util import normalize_fallacy_name
from missci.util.fileutil import read_jsonl

UNPARSEABLE_CLASS = "Other"
FALLACY_LINE_PATTERN = re.compile(r"^fallacy: (.+)$", re.IGNORECASE)


def get_fallacy_name_aliases() -> dict[str, str]:
    """Class names and spelling variants recognised in a raw answer."""
    aliases: dict[str, str] = {name: name for name in get_all_fallacy_names_from_prediction()}
    aliases["False Causality"] = "Causal Oversimplification"
    aliases["Causal Simplification"] = "Causal Oversimplification"
    aliases["False Cause"] = "Causal Oversimplification"
    aliases["Fallacy of Equivocation"] = "Ambiguity"
    aliases["Fallacy of Equivalence"] = "False Equivalence"
    aliases[UNPARSEABLE_CLASS] = UNPARSEABLE_CLASS
    return aliases


def scan_for_fallacy_name(answer: str) -> str | None:
    """The first class name occurring anywhere in the answer, if any."""
    aliases = get_fallacy_name_aliases()
    positions = sorted((answer.find(alias), alias) for alias in aliases)
    matches = [alias for position, alias in positions if position >= 0]
    if not matches:
        other_names = get_transform_other_dict()
        matches = [other_names[name] for name in other_names if name.lower() in answer.lower()]
    if not matches:
        return None
    return aliases.get(matches[0], matches[0])


def get_unique_prefix_class(fallacy_name: str) -> str | None:
    """The MISSCI class whose name starts with ``fallacy_name`` at a word boundary, if exactly one does."""
    name = re.sub(r"[\s.:;,!]+$", "", fallacy_name.strip()).lower()
    if not name:
        return None
    classes = [c for c in get_valid_fallacy_names() if c != UNPARSEABLE_CLASS]
    matches = [c for c in classes if c.lower() == name or (c.lower().startswith(name) and c.lower()[len(name)] in " /")]
    return matches[0] if len(matches) == 1 else None


def parse_answer(answer: str) -> tuple[str, str]:
    """Fallacy class and which of the three parse paths produced it."""
    lines = [line.strip() for line in answer.split("\n") if line.strip()]
    requested_format = [line for line in lines if FALLACY_LINE_PATTERN.match(line)]
    if requested_format:
        fallacy_name, parse_level = FALLACY_LINE_PATTERN.match(requested_format[0]).group(1).strip(), "strict"
    else:
        scanned = scan_for_fallacy_name(answer)
        if scanned is None:
            return UNPARSEABLE_CLASS, "none"
        fallacy_name, parse_level = scanned, "scan"
    fallacy_class = normalize_fallacy_name(fallacy_name, fail_if_unk_fallacy=False)
    fallacy_class = fallacy_class or get_unique_prefix_class(fallacy_name)
    return fallacy_class or UNPARSEABLE_CLASS, parse_level


def score_predictions(prediction_file: str) -> dict:
    """Parse and score one prediction file; the per-item lists are in file order."""
    rows = list(read_jsonl(prediction_file))
    gold = [row["data"]["gold_fallacy_class"] for row in rows]
    parsed = [parse_answer(row["answer"]) for row in rows]
    predicted = [fallacy_class for fallacy_class, _ in parsed]
    parse_levels = [parse_level for _, parse_level in parsed]
    scored = [p if p in set(gold) else "unk" for p in predicted]
    report = classification_report(gold, scored, zero_division=0, output_dict=True)
    return {
        "file": Path(prediction_file).name,
        "n": len(rows),
        "acc": sum(g == s for g, s in zip(gold, scored, strict=True)) / len(rows),
        "macro_f1": report["macro avg"]["f1-score"],
        "strict": parse_levels.count("strict"),
        "scan": parse_levels.count("scan"),
        "none": parse_levels.count("none"),
        "unparseable_class": predicted.count(UNPARSEABLE_CLASS),
        "unk": scored.count("unk"),
        "correct": [g == s for g, s in zip(gold, scored, strict=True)],
        "report": report,
        "ids": [row["data"]["fallacy_id"] for row in rows],
        "gold": gold,
        "predicted": predicted,
        "scored": scored,
    }


def get_bootstrap_ci(correct: Sequence[float], n_resamples: int, seed: int = 0) -> tuple[float, float]:
    """Percentile CI of accuracy over resampled test items."""
    rng = random.Random(seed)
    n = len(correct)
    means = sorted(sum(correct[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    return means[int(0.025 * n_resamples)], means[int(0.975 * n_resamples)]


def get_mcnemar_test(correct_a: list[bool], correct_b: list[bool]) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on paired per-item correctness."""
    only_b = sum(1 for a, b in zip(correct_a, correct_b, strict=True) if not a and b)
    only_a = sum(1 for a, b in zip(correct_a, correct_b, strict=True) if a and not b)
    discordant = only_a + only_b
    if discordant == 0:
        return only_b, only_a, 1.0
    tail = sum(comb(discordant, i) for i in range(min(only_a, only_b) + 1))
    return only_b, only_a, min(1.0, 2 * tail / 2**discordant)


def print_per_class(report: dict) -> None:
    print(f"\n{'class':<42} {'P':>6} {'R':>6} {'F1':>6} {'support':>8}")
    for class_name, scores in report.items():
        if isinstance(scores, dict) and class_name not in ("macro avg", "weighted avg"):
            print(
                f"{class_name:<42} {scores['precision']:>6.3f} {scores['recall']:>6.3f} "
                f"{scores['f1-score']:>6.3f} {scores['support']:>8.0f}"
            )


def write_csv(results: list[dict], path: str) -> None:
    import csv

    columns = ("file", "n", "acc", "macro_f1", "strict", "scan", "none", "unparseable_class", "unk")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for result in results:
            writer.writerow([result[column] for column in columns])
    print(f"\nwrote {path}")


def analyze_predictions(
    files: str,
    per_class: bool = False,
    mcnemar_with: str | None = None,
    csv_file: str | None = None,
    bootstrap_resamples: int = 10000,
) -> None:
    """Score every prediction file matching ``files`` (a path or a glob)."""
    paths = sorted(globlib.glob(files)) if any(c in files for c in "*?[") else [files]
    if not paths:
        print(f"no files matched {files}")
        raise typer.Exit(1)

    header = (
        f"{'file':<74} {'n':>5} {'acc':>6} {'mF1':>6} | "
        f"{'strict':>6} {'scan':>5} {'none':>5} {UNPARSEABLE_CLASS:>6} {'unk':>5}"
    )
    print(header)
    print("-" * len(header))
    results = []
    for path in paths:
        result = score_predictions(path)
        results.append(result)
        print(
            f"{result['file'][:74]:<74} {result['n']:>5} {result['acc']:>6.3f} {result['macro_f1']:>6.3f} | "
            f"{result['strict']:>6} {result['scan']:>5} {result['none']:>5} "
            f"{result['unparseable_class']:>6} {result['unk']:>5}"
        )

    if len(results) == 1:
        result = results[0]
        low, high = get_bootstrap_ci(result["correct"], bootstrap_resamples)
        failures = result["scan"] + result["none"]
        print(f"\naccuracy {result['acc']:.3f}, bootstrap 95% CI [{low:.3f}, {high:.3f}]")
        print(
            f"answers not in the requested format: {failures}/{result['n']} "
            f"({100 * failures / result['n']:.1f}%; recovered by scan {result['scan']}, "
            f"no class name {result['none']}); scored unk: {result['unk']}/{result['n']}"
        )
        if per_class:
            print_per_class(result["report"])
        if mcnemar_with:
            other = score_predictions(mcnemar_with)
            only_other, only_this, p_value = get_mcnemar_test(result["correct"], other["correct"])
            print(
                f"\nMcNemar vs {other['file']}: accuracy {result['acc']:.3f} vs {other['acc']:.3f}, "
                f"only-this-correct {only_this}, only-other-correct {only_other}, p = {p_value:.2g}"
            )

    if csv_file:
        write_csv(results, csv_file)


if __name__ == "__main__":
    typer.run(analyze_predictions)
