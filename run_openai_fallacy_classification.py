"""Zero-shot fallacy classification with an OpenAI model, with the prompts and output format of
run_mlx_fallacy_classification.py (temperature 0, max_tokens 1024).

An interrupted run resumes from <prediction file>.partial. Once the prediction file exists, no API calls
are made: the file is scored and each fine-tuned seed is compared with it.

Usage:
    export OPENAI_API_KEY=...
    python run_openai_fallacy_classification.py --model-name gpt-4.1
"""

import json
import os
from pathlib import Path

import openai
import typer

from aggregate_predictions import DISPLAY_NAMES
from aggregate_predictions import FULL_DATASET
from aggregate_predictions import MODELS
from aggregate_predictions import TEMPLATE
from aggregate_predictions import get_macro_f1_ci
from aggregate_predictions import get_seed_files
from aggregate_predictions import score
from analyze_predictions import get_bootstrap_ci
from analyze_predictions import get_mcnemar_test
from common import MissciSplit
from missci.util.fileutil import read_jsonl
from missci.util.fileutil import write_jsonl
from run_mlx_fallacy_classification import MAX_TOKENS
from run_mlx_fallacy_classification import MLXClassifyGenerateTemplateFiller
from run_mlx_fallacy_classification import get_prediction_filename

# As for the accuracy CIs in aggregate_predictions.py.
BOOTSTRAP_RESAMPLES = 10000

# As in the MLX runner.
REPLACE_MAP = [("Fallacy: Fallacy of Division/Composition", "Fallacy: Fallacy of Composition")]


def get_item_key(data: dict) -> str:
    return f"{data['argument']}|{data['fallacy_id']}"


def read_partial(partial_path: str) -> dict[str, dict]:
    if not os.path.exists(partial_path):
        return {}
    with open(partial_path) as f:
        results = [json.loads(line) for line in f if line.strip()]
    return {get_item_key(result["data"]): result for result in results}


def print_comparison(prediction_file: str, predictions_dir: str, eval_set: str) -> None:
    """Score the file and compare each seed file of FULL_DATASET with it (McNemar)."""
    reference = score(prediction_file)
    acc_low, acc_high = get_bootstrap_ci(reference["correct"], BOOTSTRAP_RESAMPLES)
    f1_low, f1_high = get_macro_f1_ci((prediction_file,))
    print(
        f"acc {reference['acc']:.3f} [{acc_low:.3f}, {acc_high:.3f}], "
        f"macro F1 {reference['macro_f1']:.3f} [{f1_low:.3f}, {f1_high:.3f}], unk {reference['unk']}"
    )
    if TEMPLATE not in Path(prediction_file).name:
        return
    print(f"Seed files of {FULL_DATASET} vs this file (fine-tuned-only / this-file-only correct, McNemar p):")
    for model in MODELS:
        tests = []
        for seed, path in get_seed_files(predictions_dir, model, eval_set, FULL_DATASET).items():
            if score(path)["ids"] != reference["ids"]:
                message = f"{path} does not contain the same items in the same order as {prediction_file}"
                raise ValueError(message)
            ft_only, reference_only, p_value = get_mcnemar_test(reference["correct"], score(path)["correct"])
            tests.append(f"s{seed} {ft_only}/{reference_only} p={p_value:.2g}")
        if tests:
            print(f"  {DISPLAY_NAMES[model]:<22} " + "; ".join(tests))


def run_openai_fallacy_classification(
    model_name: str = "gpt-4.1",
    prompt_template: str = "cls_with_premise/classify-D.txt",
    split: MissciSplit = MissciSplit.TEST,
    output_folder: str = "missci/predictions/classify-given-gold-premise-raw",
    seed: int = 1,
) -> None:
    dest_path = os.path.join(output_folder, get_prediction_filename(model_name, prompt_template, str(split), None))
    if os.path.exists(dest_path):
        print(f"{dest_path} exists (delete it to re-run); scoring it")
        print_comparison(dest_path, output_folder, str(split))
        return
    partial_path = f"{dest_path}.partial"

    template_filler = MLXClassifyGenerateTemplateFiller(prompt_template)
    samples = read_jsonl(f"missci/dataset/{split}.missci.jsonl")
    tasks = [task for sample in samples for task in template_filler.get_prompts(sample)]
    done = read_partial(partial_path)

    client = openai.OpenAI(max_retries=5)
    for task in tasks:
        key = get_item_key(task["data"])
        if key in done:
            continue
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": task["prompt"]}],
            temperature=0,
            max_tokens=MAX_TOKENS,
            seed=seed,
        )
        answer = response.choices[0].message.content or ""
        for replace_from, replace_to in REPLACE_MAP:
            if replace_from.lower() in answer.lower():
                answer = replace_to
        result = {
            "answer": answer,
            "params": {
                "template": prompt_template,
                "seed": seed,
                "max_tokens": MAX_TOKENS,
                "temperature": 0,
                "model": response.model,
                "system_fingerprint": response.system_fingerprint,
                "finish_reason": response.choices[0].finish_reason,
            },
            "data": task["data"],
        }
        with open(partial_path, "a") as f:
            f.write(json.dumps(result) + "\n")
        done[key] = result
        print(f"{len(done)}/{len(tasks)} {answer.splitlines()[-1] if answer.strip() else '(empty)'}")

    write_jsonl(dest_path, [done[get_item_key(task["data"])] for task in tasks])
    os.remove(partial_path)
    print(f"wrote {dest_path}")
    print_comparison(dest_path, output_folder, str(split))


if __name__ == "__main__":
    typer.run(run_openai_fallacy_classification)
