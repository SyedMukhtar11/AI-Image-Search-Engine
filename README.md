# AI-Powered Image Search Engine Using CLIP

An AI-powered image search engine that allows users to search for images using natural language descriptions. It uses the **CLIP (Contrastive Language–Image Pre-training)** model to understand the relationship between text and images and retrieve visually relevant results.

## Overview

Traditional image search systems often depend on filenames, tags, or manually assigned labels. This project uses CLIP to perform semantic image retrieval, allowing users to enter text queries and find matching images based on their visual meaning.

The system converts images and text descriptions into embeddings and compares them using cosine similarity to retrieve the most relevant results.

## Features

- Natural language-based image search.
- CLIP-based text and image understanding.
- Image embeddings using PyTorch and Hugging Face Transformers.
- Cosine similarity for image ranking.
- Top-K relevant image retrieval.
- Interactive Streamlit user interface.
- Semantic search without requiring exact filename matches.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenAI CLIP | Text-image understanding |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | CLIP model loading |
| NumPy | Embedding storage and processing |
| Scikit-learn | Cosine similarity |
| Streamlit | Web application interface |

## Model Used

**Model:** `openai/clip-vit-base-patch32`

CLIP uses two encoders:

- Image Encoder: Converts images into numerical embeddings.
- Text Encoder: Converts text queries into numerical embeddings.

Both embeddings are represented in a shared vector space, allowing the system to calculate their similarity.

## Project Architecture

1. Load images from the image dataset.
2. Process images using the CLIP image encoder.
3. Generate and store image embeddings.
4. Accept a natural language query from the user.
5. Convert the query into a text embedding using CLIP.
6. Calculate cosine similarity between text and image embeddings.
7. Rank images based on similarity scores.
8. Display the Top-K matching images through Streamlit.

## Project Structure

```text
AI-Image-Search-Engine/
│
├── app.py
├── main.py
├── .gitignore
│
├── src/
│   ├── clip_model.py
│   ├── config.py
│   ├── embeddings.py
│   ├── search.py
│   └── utils.py
│
├── data/
│   ├── images/
│   ├── image_embeddings.npy
│   └── image_filenames.npy
│
└── README.md
```

**Note:** The dataset and generated embedding files are excluded from this GitHub repository.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/AI-Image-Search-Engine.git
cd AI-Image-Search-Engine
```

### 2. Create a virtual environment

```bash
py -3.13 -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install torch transformers numpy scikit-learn streamlit pillow
```

### 5. Add your image dataset

Place your images inside:

```text
data/images/
```

### 6. Generate image embeddings

Run the image-processing script:

```bash
py -3.13 main.py
```

This generates the image embeddings and corresponding filenames required for image retrieval.

### 7. Launch the application

```bash
py -3.13 -m streamlit run app.py
```

The application will open in your browser.

## Similarity Calculation

The project uses cosine similarity to measure the similarity between text and image embeddings.

\[
\text{Similarity}(A,B)=
\frac{A\cdot B}{\|A\|\|B\|}
\]

A higher similarity score indicates a stronger semantic match between the text query and the image.

## Future Enhancements

- Support for larger image datasets.
- Improved retrieval speed using vector databases.
- Image filtering and category-based search.
- Support for multiple languages.
- Deployment as a cloud-based application.
- Improved search ranking and retrieval evaluation.

## Conclusion

This project demonstrates how Vision-Language Models can be used to build intelligent image retrieval systems. By leveraging CLIP and cosine similarity, users can search images using natural language rather than relying on filenames or manually assigned labels.

---

**Developed as part of an internship at IIT Bhubaneswar.**

**Presented by:** Syed Mukhtaruddin Quraishi