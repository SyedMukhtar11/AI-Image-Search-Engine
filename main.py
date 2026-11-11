from src.clip_model import CLIPModelManager
from src.embeddings import EmbeddingStore
from src.search import ImageSearchEngine
from src.utils import print_results


def main():

    print("=" * 60)
    print("AI IMAGE SEARCH ENGINE")
    print("=" * 60)

    # Load CLIP
    clip_model = CLIPModelManager()

    # Load stored image embeddings
    embedding_store = EmbeddingStore()

    # Create search engine
    search_engine = ImageSearchEngine(
        clip_model,
        embedding_store
    )

    print("\nSystem ready.")

    while True:

        query = input(
            "\nEnter search query "
            "(or type 'exit'): "
        ).strip()

        if query.lower() == "exit":

            print("Exiting...")
            break

        try:

            results = search_engine.search(
                query=query,
                top_k=5
            )

            print_results(results)

        except Exception as e:

            print(f"Error: {e}")


if __name__ == "__main__":
    main()