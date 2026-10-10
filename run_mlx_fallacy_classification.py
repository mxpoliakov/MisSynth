import os
from collections.abc import Iterable

import mlx
import typer
from mlx_lm import generate
from mlx_lm import load

from common import DEFAULT_SYSTEM_PROMPT
from common import MissciSplit
from missci.prompt_templates.fallacy_classify_template_filler import FallacyWiseTemplateFiller
from missci.util.fileutil import read_jsonl
from missci.util.fileutil import read_text
from missci.util.fileutil import write_jsonl

# Generation cap per answer (greedy decoding).
MAX_TOKENS = 1024


def filled_template_to_prompt(filled_template: str) -> str:
    if "@@system_prompt@@" in filled_template:
        filled_template = filled_template.replace("@@system_prompt@@", DEFAULT_SYSTEM_PROMPT)

    filled_template = filled_template.strip()

    if "@@" in filled_template:
        raise ValueError

    return filled_template


class MLXClassifyGenerateTemplateFiller(FallacyWiseTemplateFiller):
    def __init__(self, prompt_template_name: str) -> None:
        self.prompt_template_name = prompt_template_name
        self.dest_file_prefix = ""
        self.prompt_template: str = read_text(os.path.join("missci/prompt_templates", prompt_template_name))

    def get_prompts(self, argument: dict) -> Iterable[dict]:
        for item in self._get_items_for_prompt(argument):
            filled_template: str = self._fill_template(item, argument)

            yield {
                "data": self._get_base_data(argument, self._get_item_data(item, argument)),
                "prompt": filled_template_to_prompt(filled_template),
            }


def get_prediction_filename(model_name: str, prompt_template: str, dataset_name: str, adapter_path: str | None) -> str:
    """Prediction file name, unique per model, template, dataset and adapter."""
    template = prompt_template.replace("/", "_").replace(".txt", "")
    name = f"{model_name}_{template}_{dataset_name}"
    if adapter_path is not None:
        adapter_name = os.path.basename(os.path.normpath(adapter_path)).removeprefix("adapters_")
        adapter_name = adapter_name.removeprefix(f"{model_name}_")
        name += f"_{adapter_name}"
    return f"{name}.jsonl"


def query_mlx_model(
    model_name: str,
    prompt_template: str,
    dataset_name: str,
    data: list[dict],
    template_filler: MLXClassifyGenerateTemplateFiller,
    output_folder: str,
    adapter_path: str | None = None,
    seed: int = 1,
    overwrite: bool = False,
) -> None:
    mlx.core.random.seed(seed)
    model, tokenizer = load(f"mlx-community/{model_name}", adapter_path=adapter_path)
    dest_name = get_prediction_filename(model_name, prompt_template, dataset_name, adapter_path)
    dest_path = os.path.join(output_folder, dest_name)
    if os.path.exists(dest_path) and not overwrite:
        message = f"{dest_path} exists; pass --overwrite to replace it"
        raise FileExistsError(message)
    log_params: dict = {"template": prompt_template, "seed": seed, "max_tokens": MAX_TOKENS}
    replace_map = [("Fallacy: Fallacy of Division/Composition", "Fallacy: Fallacy of Composition")]
    predictions = []
    for sample in data:
        prompt_tasks = template_filler.get_prompts(sample)
        for prompt_task in prompt_tasks:
            prompt: str = prompt_task["prompt"]
            data: dict = prompt_task["data"]
            if tokenizer.chat_template is not None:
                messages = [{"role": "user", "content": prompt}]
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            answer = generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, verbose=False)
            for replace_from, replace_to in replace_map:
                if replace_from.lower() in answer.lower():
                    answer = replace_to
            result = {"answer": answer}
            print(result)
            result["params"] = log_params
            result["data"] = data
            predictions.append(result)

    write_jsonl(dest_path, predictions)


def run_mlx_fallacy_classification(
    model_name: str = "phi-4-8bit",
    prompt_template: str = "cls_with_premise/classify-D.txt",
    split: MissciSplit = MissciSplit.TEST,
    output_folder: str = "missci/predictions/classify-given-gold-premise-raw",
    adapter_path: str | None = None,
    dataset_path: str | None = None,
    overwrite: bool = False,
) -> None:
    data = list(read_jsonl(dataset_path or f"missci/dataset/{split}.missci.jsonl"))
    # e.g. "test" for the MISSCI split, "mafalda"/"logic" for the cross-dataset files
    dataset_name = os.path.basename(dataset_path).split(".")[0] if dataset_path else str(split)
    template_filler = MLXClassifyGenerateTemplateFiller(prompt_template)
    query_mlx_model(
        model_name=model_name,
        prompt_template=prompt_template,
        dataset_name=dataset_name,
        data=data,
        template_filler=template_filler,
        output_folder=output_folder,
        adapter_path=adapter_path,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    typer.run(run_mlx_fallacy_classification)
