import json
import os

import typer
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from common import DEFAULT_EMBEDDINGS_MODEL_NAME
from common import DEFAULT_VECTOR_STORE_FILENAME
from common import MissciSplit
from missci.util.fileutil import read_jsonl
from missci.util.fileutil import write_jsonl


def get_output_json(raw_output_folder: str, sample_id: str) -> list[dict]:
    try:
        with open(f"output/{raw_output_folder}/raw/{sample_id}.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def create_unified_dataset(
    embeddings_model_name: str = DEFAULT_EMBEDDINGS_MODEL_NAME,
    vector_store_filename: str = DEFAULT_VECTOR_STORE_FILENAME,
    similarity_search_k: int = 5,
    split: MissciSplit = MissciSplit.DEV,
    model_name: str = "o4-mini",
    raw_output_folders: list[str] | None = None,
) -> None:
    if raw_output_folders is None:
        raw_output_folders = [output_file for output_file in os.listdir("output") if ".jsonl" not in output_file]
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    vector_store = InMemoryVectorStore.load(f"vector_stores/{vector_store_filename}", embeddings)
    data = list(read_jsonl(f"missci/dataset/{split}.missci.jsonl"))

    dataset_list = []

    for sample in data:
        url = sample["study"]["url"]

        def filter_by_source(doc: Document, url: str = url) -> bool:
            return doc.metadata["source"] == url

        docs = vector_store.similarity_search(
            sample["argument"]["claim"], k=similarity_search_k, filter=filter_by_source
        )
        article_excerpt = "\n".join([doc.page_content for doc in docs])
        row = {
            "id": sample["id"],
            "missci_claim": sample["argument"]["claim"],
            "missci_premise": sample["argument"]["accurate_premise_p0"]["premise"],
            "rag_article_excerpt": article_excerpt,
        }

        for raw_output_folder in raw_output_folders:
            if "single-class-synthetic-fallacy" in raw_output_folder:
                row["synthetic_fallacies"] = get_output_json(raw_output_folder, sample["id"])
            elif "synthetic-claim-premise" in raw_output_folder:
                row["synthetic_claims_and_premises"] = get_output_json(raw_output_folder, sample["id"])
            else:
                message = f"{raw_output_folder} has no valid parser functions"
                raise ValueError(message)

        dataset_list.append(row)

    write_jsonl(f"dataset/MisSynth.{model_name}.jsonl", dataset_list)


if __name__ == "__main__":
    typer.run(create_unified_dataset)
