import json
import os
from copy import copy

import lorem
import typer

from common import DEFAULT_SYSTEM_PROMPT
from common import MissciSplit
from missci.util.fallacy_util import get_valid_fallacy_names
from missci.util.fallacy_util import normalize_fallacy_name
from missci.util.fileutil import read_jsonl
from missci.util.fileutil import write_jsonl


def get_additional_fallacy_mapping_dict() -> dict:
    return {"Biased Sample": "Biased Sample Fallacy"}


def get_prompt(
    prompt_template: str,
    claim: str,
    premise: str,
    context: str,
    fallacy_text: str,
) -> str:
    with open(f"missci/prompt_templates/{prompt_template}") as f:
        template_content = f.read()
    prompt = copy(template_content)
    prompt = prompt.replace("@@system_prompt@@", DEFAULT_SYSTEM_PROMPT)
    prompt = prompt.replace("@@claim@@", claim)
    prompt = prompt.replace("@@p0@@", premise)
    prompt = prompt.replace("@@context@@", context)
    prompt = prompt.replace("@@fallacious_premise@@", fallacy_text)
    return prompt


def get_output_json(raw_output_folder: str, sample_id: str) -> list[dict]:
    try:
        with open(f"output/{raw_output_folder}/raw/{sample_id}.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def generate_lorem_ipsum() -> str:
    return lorem.get_sentence(count=(1, 5))


def add_synthetic_fallacies_prompts(
    train_dataset_list: list[dict], raw_output_folder: str, prompt_template: str, sample: dict, random_baseline: bool
) -> None:
    additional_fallacy_mapping_dict = get_additional_fallacy_mapping_dict()
    valid_classes = get_valid_fallacy_names()
    output_json = get_output_json(raw_output_folder, sample["id"])
    if output_json is None:
        return
    for synthetic_entry in output_json:
        synthetic_class = synthetic_entry["class"]
        if not synthetic_class:
            continue

        fallacy_text = synthetic_entry["fallacy"] if not random_baseline else generate_lorem_ipsum()
        context = synthetic_entry["context"] if not random_baseline else generate_lorem_ipsum()
        synthetic_class = additional_fallacy_mapping_dict.get(synthetic_class, synthetic_class)
        synthetic_class = normalize_fallacy_name(synthetic_class, fail_if_unk_fallacy=False)
        if synthetic_class is None:
            continue
        assert synthetic_class in valid_classes, f"{synthetic_class} {sample['id']}"
        train_dataset_list.append(
            {
                "prompt": get_prompt(
                    prompt_template,
                    claim=sample["argument"]["claim"],
                    premise=sample["argument"]["accurate_premise_p0"]["premise"],
                    context=context,
                    fallacy_text=fallacy_text,
                ),
                "completion": f"Fallacy: {synthetic_class}",
            }
        )


def add_synthetic_claim_premise_prompts(
    train_dataset_list: list[dict], raw_output_folder: str, prompt_template: str, sample: dict, random_baseline: bool
) -> None:
    output_json = get_output_json(raw_output_folder, sample["id"])
    if output_json is None:
        return
    for fallacy in sample["argument"]["fallacies"]:
        for interchangeable_fallacy in fallacy["interchangeable_fallacies"]:
            for synthetic_entry in output_json:
                claim = synthetic_entry["claim"] if not random_baseline else generate_lorem_ipsum()
                premise = synthetic_entry["premise"] if not random_baseline else generate_lorem_ipsum()
                train_dataset_list.append(
                    {
                        "prompt": get_prompt(
                            prompt_template,
                            claim=claim,
                            premise=premise,
                            context=fallacy.get("context", ""),
                            fallacy_text=interchangeable_fallacy["premise"],
                        ),
                        "completion": f"Fallacy: {interchangeable_fallacy['class']}",
                    }
                )


def create_fine_tuning_dataset(
    prompt_template: str = "cls_with_premise/classify-D.txt",
    split: MissciSplit = MissciSplit.DEV,
    raw_output_folders: list[str] | None = None,
    random_baseline: bool = False,
) -> None:
    if raw_output_folders is None:
        raw_output_folders = [output_file for output_file in os.listdir("output") if ".jsonl" not in output_file]

    data = list(read_jsonl(f"missci/dataset/{split}.missci.jsonl"))

    train_dataset_list = []
    val_dataset_list = []

    for sample in data:
        for raw_output_folder in raw_output_folders:
            if "single-class-synthetic-fallacy" in raw_output_folder:
                add_synthetic_fallacies_prompts(
                    train_dataset_list, raw_output_folder, prompt_template, sample, random_baseline
                )
            elif "synthetic-claim-premise" in raw_output_folder:
                add_synthetic_claim_premise_prompts(
                    train_dataset_list, raw_output_folder, prompt_template, sample, random_baseline
                )
            else:
                message = f"{raw_output_folder} has no valid parser functions"
                raise ValueError(message)
        for fallacy in sample["argument"]["fallacies"]:
            for interchangeable_fallacy in fallacy["interchangeable_fallacies"]:
                val_dataset_list.append(
                    {
                        "prompt": get_prompt(
                            prompt_template,
                            claim=sample["argument"]["claim"],
                            premise=sample["argument"]["accurate_premise_p0"]["premise"],
                            context=fallacy.get("context", ""),
                            fallacy_text=interchangeable_fallacy["premise"],
                        ),
                        "completion": f"Fallacy: {interchangeable_fallacy['class']}",
                    }
                )
    print(f"Train samples: {len(train_dataset_list)}. Val samples: {len(val_dataset_list)}")
    write_jsonl("output/train.jsonl", train_dataset_list)
    write_jsonl("output/valid.jsonl", val_dataset_list)


if __name__ == "__main__":
    typer.run(create_fine_tuning_dataset)
