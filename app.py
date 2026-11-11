
import streamlit as st
from PIL import Image

from src.clip_model import CLIPModelManager
from src.embeddings import EmbeddingStore
from src.search import ImageSearchEngine
from src.utils import get_image_path


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Image Search Engine",
    page_icon="🔎",
    layout="wide"
)


# --------------------------------------------------
# Custom styling
# --------------------------------------------------

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .hero {
        padding: 1.8rem;
        border-radius: 18px;
        background: linear-gradient(
            120deg,
            #142d4e,
            #245b75
        );
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: white;
        margin-bottom: 0.4rem;
    }

    .hero p {
        color: #e1edf5;
        font-size: 1.05rem;
    }

    .result-card {
        padding: 0.8rem;
        border: 1px solid #d8e1e8;
        border-radius: 12px;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Load backend once and cache it
# --------------------------------------------------

@st.cache_resource
def load_search_engine():

    clip_model = CLIPModelManager()
    embedding_store = EmbeddingStore()

    engine = ImageSearchEngine(
        clip_model,
        embedding_store
    )

    return engine


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>🔎 AI Image Search Engine</h1>
    <p>
        Discover images using natural language.
        Powered by CLIP and semantic similarity search.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.title("Search Settings")

    top_k = st.slider(
        "Number of results",
        min_value=1,
        max_value=10,
        value=5
    )

    st.divider()

    st.markdown("### How it works")
    st.write("1. Enter a text query.")
    st.write("2. CLIP converts text into an embedding.")
    st.write("3. Cosine similarity ranks your images.")
    st.write("4. The most relevant images are displayed.")

    st.caption("Model: CLIP ViT-B/32")


# --------------------------------------------------
# Query input
# --------------------------------------------------

query = st.text_input(
    "Describe the image you want to find",
    placeholder="e.g. a dog playing outside"
)

search_clicked = st.button(
    "🔍 Search Images",
    type="primary",
    use_container_width=True
)


if search_clicked:

    if not query.strip():
        st.warning("Please enter a search query.")

    else:
        try:
            with st.spinner(
                "Loading model and searching images..."
            ):
                engine = load_search_engine()

                results = engine.search(
                    query=query,
                    top_k=top_k
                )

            if not results:
                st.info("No results were returned.")

            else:
                st.success(
                    f"Found {len(results)} results "
                    f"for: {query}"
                )

                st.subheader("Search Results")

                columns = st.columns( min(len(results), 5) )

                for i, result in enumerate(results):

                    with columns[i % len(columns)]:

                        try:
                            image_path = get_image_path(
                                result["filename"]
                            )

                            with Image.open(image_path) as img:
                                display_image = img.convert("RGB")

                            st.image(
                                display_image,
                                use_container_width=True
                            )

                            st.markdown(
                                f"**Rank #{result['rank']}**"
                            )

                            st.caption(
                                result["filename"]
                            )

                           

                            

                        except (FileNotFoundError, OSError) as error:
                            st.warning(
                                f"Cannot display "
                                f"{result['filename']}: {error}"
                            )

        except Exception as error:
            st.error(f"Search failed: {error}")
            st.info(
                "Check your embedding files, image paths, "
                "model loading, and backend configuration."
            )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "AI Image Search Engine | CLIP + PyTorch + Streamlit"
)