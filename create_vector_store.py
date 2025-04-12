import bs4
import typer
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf.errors import PdfStreamError
from requests.exceptions import ConnectTimeout

from common import DEFAULT_EMBEDDINGS_MODEL_NAME
from common import DEFAULT_VECTOR_STORE_FILENAME
from common import MissciDataset
from missci.util.fileutil import read_jsonl


def create_vector_store(
    embeddings_model_name: str = DEFAULT_EMBEDDINGS_MODEL_NAME,
    dataset: MissciDataset = MissciDataset.TEST,
    vector_store_filename: str = DEFAULT_VECTOR_STORE_FILENAME,
    min_page_content_length: int = 1000,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> None:
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    vector_store = InMemoryVectorStore(embeddings)
    bs4_strainer = bs4.SoupStrainer(["p"])
    docs = []
    valid_samples_count = 0
    data = list(read_jsonl(f"missci/dataset/{dataset}.missci.jsonl"))
    for sample in data:
        url = sample["study"]["url"]
        if "pdf" in url:
            try:
                loader = PyPDFLoader(url, mode="single")
            except (ConnectTimeout, ValueError):
                continue
        else:
            loader = WebBaseLoader(url, bs_kwargs={"parse_only": bs4_strainer})

        try:
            for doc in loader.lazy_load():
                if len(doc.page_content) > min_page_content_length:
                    print(f"Loaded url: {url}")
                    docs.append(doc)
                    valid_samples_count += 1
                    break
        except PdfStreamError:
            continue

    print(f"Loaded {valid_samples_count} out of {len(data)}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_splits = text_splitter.split_documents(docs)

    vector_store.add_documents(documents=all_splits)
    vector_store.dump(f"vector_stores/{vector_store_filename}")


if __name__ == "__main__":
    typer.run(create_vector_store)
