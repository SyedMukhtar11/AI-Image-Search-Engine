import numpy as np

from .config import (
    EMBEDDING_FILE,
    FILENAME_FILE
)


class EmbeddingStore:

    def __init__(self):

        self.image_embeddings = np.load(
            EMBEDDING_FILE
        )

        self.image_filenames = np.load(
            FILENAME_FILE
        )

        if len(self.image_embeddings) != len(
            self.image_filenames
        ):
            raise ValueError(
                "Number of embeddings does not "
                "match number of filenames."
            )

        print(
            f"Loaded {len(self.image_embeddings)} "
            "image embeddings."
        )

        print(
            "Embedding shape:",
            self.image_embeddings.shape
        )

    def get_embeddings(self):
        return self.image_embeddings

    def get_filenames(self):
        return self.image_filenames