
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ImageSearchEngine:

    def __init__(self, clip_model, embedding_store):
        self.clip_model = clip_model
        self.embedding_store = embedding_store

        self.image_embeddings = np.asarray(
            embedding_store.get_embeddings(),
            dtype=np.float32
        )

        self.image_filenames = np.asarray(
            embedding_store.get_filenames()
        )

        # Validate the saved data
        if len(self.image_embeddings) != len(self.image_filenames):
            raise ValueError(
                "Mismatch: image embeddings and filenames "
                "have different lengths."
            )

        if self.image_embeddings.ndim != 2:
            raise ValueError("Image embeddings must be a 2D array.")

        # Normalize image embeddings
        norms = np.linalg.norm(
            self.image_embeddings, axis=1, keepdims=True
        )
        self.image_embeddings = self.image_embeddings / np.maximum(
            norms, 1e-12
        )

    def search(self, query, top_k=5):

        text_embedding = np.asarray(
            self.clip_model.get_text_embedding(query),
            dtype=np.float32
        ).reshape(1, -1)

        if text_embedding.shape[1] != self.image_embeddings.shape[1]:
            raise ValueError(
                f"Embedding dimension mismatch: text has "
                f"{text_embedding.shape[1]}, images have "
                f"{self.image_embeddings.shape[1]}. "
                "Regenerate the image embeddings using the same CLIP model."
            )

        # Normalize text embedding
        text_embedding /= max(
            np.linalg.norm(text_embedding), 1e-12
        )

        scores = cosine_similarity(
            text_embedding, self.image_embeddings
        )[0]

        indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for rank, i in enumerate(indices, start=1):
            score = float(scores[i])

            print(
                f"Rank {rank}: {self.image_filenames[i]} "
                f"| Cosine similarity: {score:.4f}"
            )

            results.append({
                "rank": rank,
                "filename": str(self.image_filenames[i]),
                "similarity": score
            })

        return results