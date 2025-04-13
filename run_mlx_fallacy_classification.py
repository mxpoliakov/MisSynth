import os
from collections.abc import Iterable

import mlx
import typer
from mlx_lm import generate
from mlx_lm import load

from common import DEFAULT_SYSTEM_PROMPT
from common import MissciSplit
from missci.prompt_templates.classify_generate_template_filler import ClassifyGenerateTemplateFiller
from missci.util.fileutil import read_jsonl
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
    split: MissciSplit,
    data: list[dict],
    template_filler: MLXClassifyGenerateTemplateFiller,
    output_folder: str,
    adapter_path: str | None = None,
    seed: int = 1,
) -> None:
    mlx.core.random.seed(seed)
    model, tokenizer = load(f"mlx-community/{model_name}", adapter_path=adapter_path)
    dest_base_name = f"{model_name}_{prompt_template.replace('/', '_').replace('.txt', '')}_{split}"
    if adapter_path is not None:
        dest_base_name += f"_{adapter_path}"
    dest_name = f"{dest_base_name}.jsonl"
    log_params: dict = {"template": prompt_template, "seed": seed}
    replace_map = [("Fallacy of Division/Composition", "Fallacy of Composition")]
    predictions = []
    for sample in data:
        prompt_tasks = template_filler.get_prompts(sample)
        for prompt_task in prompt_tasks:
            prompt: str = prompt_task["prompt"]
            data: dict = prompt_task["data"]
            if tokenizer.chat_template is not None:
                messages = [{"role": "user", "content": prompt}]
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            answer = generate(model, tokenizer, prompt=prompt, verbose=False)
            for replace_from, replace_to in replace_map:
                answer = answer.replace(replace_from, replace_to)
            result = {"answer": answer}
            print(result)
            result["params"] = log_params
            result["data"] = data
            predictions.append(result)

    write_jsonl(os.path.join(output_folder, dest_name), predictions)


def run_mlx_fallacy_classification(
    model_name: str = "phi-4-8bit",
    prompt_template: str = "cls_without_premise/p4-connect-cls-D.txt",
    split: MissciSplit = MissciSplit.DEV,
    output_folder: str = "missci/predictions/only-classify-raw",
    adapter_path: str | None = None,
) -> None:
    data = list(read_jsonl(f"missci/dataset/{split}.missci.jsonl"))
    template_filler = MLXClassifyGenerateTemplateFiller(prompt_template)
    query_mlx_model(
        model_name=model_name,
        prompt_template=prompt_template,
        split=split,
        data=data,
        template_filler=template_filler,
        output_folder=output_folder,
        adapter_path=adapter_path,
    )


if __name__ == "__main__":
    typer.run(run_mlx_fallacy_classification)
