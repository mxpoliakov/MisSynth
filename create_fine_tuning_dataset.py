import re
from copy import copy

import typer
from sklearn.model_selection import train_test_split

from common import DEFAULT_SYSTEM_PROMPT
from common import MissciSplit
from missci.util.fallacy_util import get_valid_fallacy_names
from missci.util.fallacy_util import normalize_fallacy_name
from missci.util.fileutil import read_jsonl
from missci.util.fileutil import write_jsonl


def get_additional_fallacy_mapping_dict() -> dict:
    return {
        "Biased Sample": "Biased Sample Fallacy",
        "Appeal to Extremes": "Impossible Expectations",
        "Slippery Slope": "Hasty Generalization",
        "Cherry Picking": "Hasty Generalization",
    }


def get_prompt(
    prompt_template: str,
    claim: str,
    premise: str,
    fallacy_text: str,
) -> str:
    with open(f"missci/prompt_templates/{prompt_template}") as f:
        template_content = f.read()
    prompt = copy(template_content)
    prompt = prompt.replace("@@system_prompt@@", DEFAULT_SYSTEM_PROMPT)
    prompt = prompt.replace("@@claim@@", claim)
    prompt = prompt.replace("@@p0@@", premise)
    prompt = prompt.replace("@@context@@", fallacy_text)
    return prompt


def create_fine_tuning_dataset(
    prompt_template: str = "cls_without_premise/p4-connect-cls-D.txt",
    split: MissciSplit = MissciSplit.TEST,
    raw_output_folder: str = "o3-mini-single-class-synthetic-fallacy-20",
    test_split_size: float = 0.15,
) -> None:
    valid_classes = get_valid_fallacy_names()
    pattern = re.compile(r"Synthetic Fallacy \d+\.\s*(?:Class\s*[-–:]?\s*)?([A-Za-z ]+)\s*[:–-]\s*(.+)")
    data = list(read_jsonl(f"missci/dataset/{split}.missci.jsonl"))
    additional_fallacy_mapping_dict = get_additional_fallacy_mapping_dict()
    fine_tuning_dataset_list = []
    for sample in data:
        try:
            with open(f"output/{raw_output_folder}/raw/{sample['id']}.txt") as f:
                result = f.read()
        except FileNotFoundError:
            continue
        matches = pattern.findall(result)
        for match in matches:
            synthetic_class = match[0].strip()
            if not synthetic_class:
                continue
            fallacy_text = match[1].strip()

            synthetic_class = additional_fallacy_mapping_dict.get(synthetic_class, synthetic_class)
            synthetic_class = normalize_fallacy_name(synthetic_class, fail_if_unk_fallacy=False)
            if synthetic_class is None:
                continue
            assert synthetic_class in valid_classes, f"{synthetic_class} {sample['id']}"
            prompt = get_prompt(
                prompt_template,
                claim=sample["argument"]["claim"],
                premise=sample["argument"]["accurate_premise_p0"]["premise"],
                fallacy_text=fallacy_text,
            )
            fine_tuning_dataset_list.append({"prompt": prompt, "completion": f"Fallacy: {synthetic_class}"})

    dataset_train, dataset_val = train_test_split(fine_tuning_dataset_list, test_size=test_split_size, random_state=1)
    print(f"Train samples: {len(dataset_train)}. Val samples: {len(dataset_val)}")
    write_jsonl(f"output/{raw_output_folder}//train.jsonl", dataset_train)
    write_jsonl(f"output/{raw_output_folder}//valid.jsonl", dataset_val)


if __name__ == "__main__":
    typer.run(create_fine_tuning_dataset)
