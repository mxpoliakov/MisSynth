import random

import typer

from common import MissciSplit
from create_fine_tuning_dataset import get_additional_fallacy_mapping_dict
from missci.util.fallacy_util import get_valid_fallacy_names
from missci.util.fallacy_util import normalize_fallacy_name
from missci.util.fileutil import read_jsonl
from missci.util.fileutil import write_jsonl


def get_training_class(synthetic_class: str) -> str | None:
    """Training label of a synthetic class, or None if create_fine_tuning_dataset.py drops it."""
    if not synthetic_class:
        return None
    synthetic_class = get_additional_fallacy_mapping_dict().get(synthetic_class, synthetic_class)
    training_class = normalize_fallacy_name(synthetic_class, fail_if_unk_fallacy=False)
    return training_class if training_class in get_valid_fallacy_names() else None


def get_gold_premises(split: MissciSplit) -> set[str]:
    return {
        " ".join(interchangeable_fallacy["premise"].lower().split())
        for sample in read_jsonl(f"missci/dataset/{split}.missci.jsonl")
        for fallacy in sample["argument"]["fallacies"]
        for interchangeable_fallacy in fallacy["interchangeable_fallacies"]
    }


def create_human_eval_dataset(
    model_name: str = "o4-mini",
    n_samples: int = 120,
    seed: int = 0,
    split: MissciSplit = MissciSplit.DEV,
) -> None:
    """Sample synthetic fallacies of the training data for the human validation, stratified by class.

    Claim/premise pairs and premises that repeat a gold dev premise are left out.
    """
    gold_premises = get_gold_premises(split)
    pool = []
    n_dropped_class = 0
    n_dropped_gold = 0
    for row in read_jsonl(f"dataset/MisSynth.{model_name}.jsonl"):
        for i, synthetic_entry in enumerate(row["synthetic_fallacies"] or []):
            training_class = get_training_class(synthetic_entry["class"])
            if training_class is None:
                n_dropped_class += 1
                continue
            if " ".join(synthetic_entry["fallacy"].lower().split()) in gold_premises:
                n_dropped_gold += 1
                continue
            pool.append(
                {
                    "id": f"{row['id']}#{i:02d}",
                    "argument_id": row["id"],
                    "missci_claim": row["missci_claim"],
                    "missci_premise": row["missci_premise"],
                    "rag_article_excerpt": row["rag_article_excerpt"],
                    "context": synthetic_entry["context"],
                    "fallacy": synthetic_entry["fallacy"],
                    "class": synthetic_entry["class"],
                    "training_class": training_class,
                }
            )

    pool_by_class = {}
    for entry in pool:
        pool_by_class.setdefault(entry["training_class"], []).append(entry)
    classes = sorted(pool_by_class)
    rng = random.Random(seed)
    per_class = dict.fromkeys(classes, n_samples // len(classes))
    for training_class in rng.sample(classes, n_samples % len(classes)):
        per_class[training_class] += 1
    sample = []
    for training_class in classes:
        sample += rng.sample(pool_by_class[training_class], per_class[training_class])
    rng.shuffle(sample)

    output_path = f"human_validation/MisSynth.{model_name}.human_eval.jsonl"
    write_jsonl(output_path, sample)
    print(
        f"Pool: {len(pool)} synthetic fallacies ({n_dropped_class} dropped for class, "
        f"{n_dropped_gold} as gold dev copies). Sampled {len(sample)} with seed {seed} -> {output_path}"
    )
    for training_class in classes:
        print(f"  {training_class}: {per_class[training_class]} of {len(pool_by_class[training_class])}")


if __name__ == "__main__":
    typer.run(create_human_eval_dataset)
