from pathlib import Path

from .config import IMAGE_DIR


def get_image_path(filename):

    image_path = IMAGE_DIR / filename

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    return image_path


def print_results(results):

    print("\n" + "=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)

    for result in results:

        print(
            f"{result['rank']}. "
            f"{result['filename']} "
            f"| Similarity: "
            f"{result['similarity']:.4f}"
        )

    print("=" * 60)