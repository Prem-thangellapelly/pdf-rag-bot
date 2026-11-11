from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json
import os
import time


# Locate the project folder.
project_folder = Path(__file__).resolve().parent

# Load your Gemini key.
load_dotenv(project_folder / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise SystemExit("Add GEMINI_API_KEY to your .env file.")


# Define the input and output paths.
input_path = project_folder / "data" / "chunks.json"
output_path = project_folder / "data" / "embeddings.json"

if not input_path.exists():
    raise SystemExit("Chunks not found. Run chunk.py first.")


# Read the chunks.
with open(input_path, "r", encoding="utf-8") as file:
    chunks = json.load(file)

if not chunks:
    raise SystemExit("The chunks file is empty.")


embedding_model = "gemini-embedding-001"
vector_dimensions = 3072
batch_size = 10

embedded_chunks = []


# Open the Gemini client.
with genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=60000)
) as client:

    for start in range(0, len(chunks), batch_size):

        batch = chunks[start:start + batch_size]

        texts = [chunk["text"] for chunk in batch]

        # Generate embeddings for document passages.
        response = client.models.embed_content(
            model=embedding_model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=vector_dimensions
            )
        )

        embeddings = response.embeddings

        # Confirm that every chunk received a vector.
        if not embeddings or len(embeddings) != len(batch):
            raise SystemExit("Unexpected embedding count. Output not saved.")

        for chunk, embedding in zip(batch, embeddings):

            vector = embedding.values

            if not vector or len(vector) != vector_dimensions:
                raise SystemExit("Unexpected vector size. Output not saved.")

            embedded_chunk = {
                "id": chunk["id"],
                "source": chunk["source"],
                "page": chunk["page"],
                "text": chunk["text"],
                "vector": vector
            }

            embedded_chunks.append(embedded_chunk)

        print(
            f"Embedded {len(embedded_chunks)} "
            f"of {len(chunks)} chunks"
        )

        # Pause between batches to reduce request frequency.
        if start + batch_size < len(chunks):
            time.sleep(7)


# Save model information with the embedded chunks.
output_data = {
    "provider": "gemini",
    "model": embedding_model,
    "dimensions": vector_dimensions,
    "records": embedded_chunks
}

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(output_data, file, indent=4, ensure_ascii=False)


print(f"\nVector dimensions: {vector_dimensions}")
print(f"Saved embeddings to: {output_path}")