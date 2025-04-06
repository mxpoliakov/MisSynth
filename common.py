from enum import StrEnum

DEFAULT_EMBEDDINGS_MODEL_NAME = "NeuML/pubmedbert-base-embeddings"
DEFAULT_VECTOR_STORE_FILENAME = "vector_store.json"


class MissciDataset(StrEnum):
    DEV = "dev"
    TEST = "test"
