import os
from collections.abc import Iterable

import mlx
import typer
from mlx_lm import generate
from mlx_lm import load

from common import DEFAULT_SYSTEM_PROMPT
from common import MissciDataset
from missci.data.missci_data_loader import MissciDataLoader
from missci.prompt_templates.classify_generate_template_filler import ClassifyGenerateTemplateFiller
from missci.util.fileutil import read_text
from missci.util.fileutil import write_jsonl


def filled_template_to_prompt(filled_template: str) -> str:
    if "@@system_prompt@@" in filled_template:
        filled_template = filled_template.replace("@@system_prompt@@", DEFAULT_SYSTEM_PROMPT)

    filled_template = filled_template.strip()

    if "@@" in filled_template:
        raise ValueError

    return filled_template


class MLXClassifyGenerateTemplateFiller(ClassifyGenerateTemplateFiller):
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


def query_mlx_model(
    model_name: str,
    prompt_template: str,
    dataset: MissciDataset,
    instances: list[dict],
    template_filler: MLXClassifyGenerateTemplateFiller,
    output_folder: str,
    seed: int = 1,
) -> None:
    mlx.core.random.seed(seed)
    model, tokenizer = load(f"mlx-community/{model_name}", adapter_path=None)

    dest_name = f"{model_name}_{prompt_template.replace('/', '_').replace('.txt', '')}_{dataset}.jsonl"
    log_params: dict = {"template": prompt_template, "seed": seed}

    predictions = []
    for argument in instances:
        prompt_tasks = template_filler.get_prompts(argument)
        for prompt_task in prompt_tasks:
            prompt: str = prompt_task["prompt"]
            data: dict = prompt_task["data"]
            if tokenizer.chat_template is not None:
                messages = [{"role": "user", "content": prompt}]
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            answer = generate(model, tokenizer, prompt=prompt, verbose=False)
            result = {"answer": answer}
            print(result)
            result["params"] = log_params
            result["data"] = data
            predictions.append(result)

    write_jsonl(os.path.join(output_folder, dest_name), predictions)


def run_mlx_fallacy_classification(
    model_name: str = "phi-4-8bit",
    prompt_template: str = "cls_without_premise/p4-connect-cls-D.txt",
    dataset: MissciDataset = MissciDataset.DEV,
    output_folder: str = "missci/predictions/only-classify-raw",
) -> None:
    instances = MissciDataLoader("missci/dataset").load_raw_arguments(dataset)
    template_filler = MLXClassifyGenerateTemplateFiller(prompt_template)
    query_mlx_model(
        model_name=model_name,
        prompt_template=prompt_template,
        dataset=dataset,
        instances=instances,
        template_filler=template_filler,
        output_folder=output_folder,
    )


if __name__ == "__main__":
    typer.run(run_mlx_fallacy_classification)
