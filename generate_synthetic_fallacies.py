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
    n_synthetic_fallacies: int,
) -> str:
    with open(f"prompt_templates/{prompt_template}.txt") as f:
        template_content = f.read()

    fallacy_inventory = get_fallacy_inventory()

    return PromptTemplate.from_template(template_content).format(
        claim=claim,
        premise=premise,
        fallacies=fallacies,
        article_excerpt=article_excerpt,
        n_synthetic_fallacies=n_synthetic_fallacies,
        fallacy_inventory=fallacy_inventory,
    )


def get_real_world_fallacies(argument_fallacies: list[dict]) -> str:
    fallacies = ""
    for i, fallacy in enumerate(argument_fallacies):
        for interchangeable_fallacy in fallacy["interchangeable_fallacies"]:
            fallacy_class = interchangeable_fallacy["class"]
            fallacy_premise = interchangeable_fallacy["premise"]
            fallacies += f"\nReal-world fallacy {i + 1}. Class - {fallacy_class}:\n{fallacy_premise}\n"
    return fallacies


def generate_synthetic_fallacies(
    embeddings_model_name: str = DEFAULT_EMBEDDINGS_MODEL_NAME,
    split: MissciSplit = MissciSplit.TEST,
    vector_store_filename: str = DEFAULT_VECTOR_STORE_FILENAME,
    model_name: str = "o3-mini",
    prompt_template: str = "single-class-synthetic-fallacy",
    similarity_search_k: int = 5,
    n_synthetic_fallacies: int = 20,
    temperature: float = 1.0,
) -> None:
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    vector_store = InMemoryVectorStore.load(f"vector_stores/{vector_store_filename}", embeddings)
    model = init_chat_model(model_name, temperature=temperature)

    output_path = Path(f"output/{model_name}-{prompt_template}-{n_synthetic_fallacies}/raw")
    output_path.mkdir(parents=True, exist_ok=True)

    for sample in read_jsonl(f"missci/dataset/{split}.missci.jsonl"):
        url = sample["study"]["url"]
        argument = sample["argument"]
        fallacies = get_real_world_fallacies(argument["fallacies"])

        def filter_by_source(doc: Document, url: str = url) -> bool:
            return doc.metadata["source"] == url

        docs = vector_store.similarity_search(argument["claim"], k=similarity_search_k, filter=filter_by_source)

        if docs:
            article_excerpt = "\n".join([doc.page_content for doc in docs])
            prompt = get_prompt(
                prompt_template=prompt_template,
                claim=argument["claim"],
                premise=argument["accurate_premise_p0"]["premise"],
                article_excerpt=article_excerpt,
                fallacies=fallacies,
                n_synthetic_fallacies=n_synthetic_fallacies,
            )
            response = model.invoke(prompt)
            with open(output_path / f"{sample['id']}.txt", "w") as f:
                f.write(response.content)


if __name__ == "__main__":
    typer.run(generate_synthetic_fallacies)
