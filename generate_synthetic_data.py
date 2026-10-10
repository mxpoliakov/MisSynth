import json
import random
import re
from pathlib import Path

import typer
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from common import DEFAULT_EMBEDDINGS_MODEL_NAME
from common import DEFAULT_VECTOR_STORE_FILENAME
from common import MissciSplit
from missci.util.fileutil import read_jsonl


def clean_json_string(string: str) -> str:
    string = string.strip()

    if string.startswith("```json") and string.endswith("```"):
        string = re.sub(r"^```json\s*", "", string)
        string = re.sub(r"\s*```$", "", string)

    return string.strip()


def get_fallacy_inventory() -> str:
    with open("missci/prompt_templates/gen_cls/p1-basic-D.txt") as f:
        text = f.read()
    start_index = text.find("Fallacy Inventory")
    end_index = text.find("Task")
    return text[start_index:end_index].strip()


def get_prompt(
    prompt_template: str,
    claim: str,
    premise: str,
    fallacies: str,
    article_excerpt: str,
    n_synthetic_entries: int,
) -> str:
    with open(f"prompt_templates/{prompt_template}.txt") as f:
        template_content = f.read()

    fallacy_inventory = get_fallacy_inventory()

    return PromptTemplate.from_template(template_content).format(
        claim=claim,
        premise=premise,
        fallacies=fallacies,
        article_excerpt=article_excerpt,
        n_synthetic_entries=n_synthetic_entries,
        fallacy_inventory=fallacy_inventory,
    )


def get_real_world_fallacies(argument_fallacies: list[dict]) -> str:
    fallacies = ""
    for i, fallacy in enumerate(argument_fallacies):
        for interchangeable_fallacy in fallacy["interchangeable_fallacies"]:
            fallacy_class = interchangeable_fallacy["class"]
            fallacy_premise = interchangeable_fallacy["premise"]
            context = fallacy.get("fallacy_context", "")
            fallacies += (
                f"\nContext {i + 1}: {context}\nFallacy {i + 1}: {fallacy_premise}\nClass {i + 1}: {fallacy_class}\n\n"
            )
    return fallacies


def get_random_source_ids(samples: list[dict], seed: int) -> dict[str, str]:
    """Map every argument id to a different argument whose article supplies the excerpt (seeded derangement)."""
    ids = [sample["id"] for sample in samples]
    urls = [sample["study"]["url"] for sample in samples]
    assert len(set(urls)) == len(urls), "two arguments share a source article"
    sources = ids[:]
    rng = random.Random(seed)
    while True:
        rng.shuffle(sources)
        if all(a != b for a, b in zip(ids, sources)):
            return dict(zip(ids, sources))


def generate_synthetic_data(
    embeddings_model_name: str = DEFAULT_EMBEDDINGS_MODEL_NAME,
    split: MissciSplit = MissciSplit.DEV,
    vector_store_filename: str = DEFAULT_VECTOR_STORE_FILENAME,
    model_name: str = "o4-mini",
    model_provider: str | None = "openai",
    prompt_template: str = "single-class-synthetic-fallacy-context",
    similarity_search_k: int = 5,
    n_synthetic_entries: int = 30,
    temperature: float = 1.0,
    max_attempts: int = 3,
    no_retrieval: bool = False,
    random_source: bool = False,
    random_source_seed: int = 0,
    dry_run: bool = False,
) -> None:
    """Generate synthetic entries for every argument of ``split``; existing outputs are skipped.

    ``no_retrieval`` omits the article excerpt, and ``random_source`` takes it from another dev article.
    """
    assert not (no_retrieval and random_source), "--no-retrieval and --random-source are separate ablations"
    suffix = "-no-retrieval" if no_retrieval else "-random-source" if random_source else ""
    template_name = f"{prompt_template}-no-retrieval" if no_retrieval else prompt_template

    vector_store = None
    if not no_retrieval:
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        vector_store = InMemoryVectorStore.load(f"vector_stores/{vector_store_filename}", embeddings)
    model = None if dry_run else init_chat_model(model_name, model_provider=model_provider, temperature=temperature)

    output_dir = Path(f"output/{model_name}-{prompt_template}{suffix}-{n_synthetic_entries}")
    output_path = output_dir / "raw"
    prompt_path = output_dir / "prompts"
    output_path.mkdir(parents=True, exist_ok=True)
    prompt_path.mkdir(parents=True, exist_ok=True)

    samples = list(read_jsonl(f"missci/dataset/{split}.missci.jsonl"))
    samples_by_id = {sample["id"]: sample for sample in samples}
    source_ids = {sample["id"]: sample["id"] for sample in samples}
    if random_source:
        source_ids = get_random_source_ids(samples, random_source_seed)
        with open(output_dir / "random_source_map.json", "w") as f:
            source_map = {
                arg_id: {"source_id": src_id, "source_url": samples_by_id[src_id]["study"]["url"]}
                for arg_id, src_id in source_ids.items()
            }
            json.dump({"seed": random_source_seed, "map": source_map}, f, indent=4)

    failed_ids = []
    for sample in samples:
        if (output_path / f"{sample['id']}.json").exists():
            continue
        url = samples_by_id[source_ids[sample["id"]]]["study"]["url"]
        argument = sample["argument"]
        fallacies = get_real_world_fallacies(argument["fallacies"])

        def filter_by_source(doc: Document, url: str = url) -> bool:
            return doc.metadata["source"] == url

        article_excerpt = ""
        if not no_retrieval:
            docs = vector_store.similarity_search(argument["claim"], k=similarity_search_k, filter=filter_by_source)
            if not docs:
                print(f"{sample['id']}: no article excerpt retrieved, skipped")
                failed_ids.append(sample["id"])
                continue
            article_excerpt = "\n".join([doc.page_content for doc in docs])
        prompt = get_prompt(
            prompt_template=template_name,
            claim=argument["claim"],
            premise=argument["accurate_premise_p0"]["premise"],
            article_excerpt=article_excerpt,
            fallacies=fallacies,
            n_synthetic_entries=n_synthetic_entries,
        )
        (prompt_path / f"{sample['id']}.txt").write_text(prompt)
        if dry_run:
            continue
        for attempt in range(1, max_attempts + 1):
            response = model.invoke(prompt)
            try:
                response_json = json.loads(clean_json_string(response.content))
            except json.JSONDecodeError:
                print(f"{sample['id']}: invalid JSON (attempt {attempt}/{max_attempts})")
                continue
            with open(output_path / f"{sample['id']}.json", "w") as f:
                json.dump(response_json, f, indent=4)
            break
        else:
            failed_ids.append(sample["id"])
    if failed_ids:
        print(f"{len(failed_ids)} argument(s) without output: {', '.join(failed_ids)}")


if __name__ == "__main__":
    typer.run(generate_synthetic_data)
