from enum import StrEnum

DEFAULT_EMBEDDINGS_MODEL_NAME = "NeuML/pubmedbert-base-embeddings"
DEFAULT_VECTOR_STORE_FILENAME = "vector_store.json"

DEFAULT_SYSTEM_PROMPT = """
Follow user instructions exactly.
Provide only what is requested—no explanations, commentary, or extra information.
Do not rephrase or expand unless specifically asked.
Answer in the most direct and concise way possible.
"""


class MissciSplit(StrEnum):
    DEV = "dev"
    TEST = "test"
