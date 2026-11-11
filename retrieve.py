from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
import json
import os


# Locate the project folder.
project_folder = Path(__file__).resolve().parent

metadata_path = project_folder / "data" / "index_meta.json"
output_path = project_folder / "data" / "retrieved_context.json"


# Load your Gemini API key.
load_dotenv(project_folder / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise SystemExit("Add GEMINI_API_KEY to your .env file.")


# Load the settings saved by store.py.
if not metadata_path.exists():
    raise SystemExit("Index metadata not found. Run store.py first.")

with open(metadata_path, "r", encoding="utf-8") as file:
    metadata = json.load(file)


# Get the user's question.
question = input("Ask a question about your PDF: ").strip()

if not question:
    raise SystemExit("Please enter a question.")

top_k = 3


# Connect to Qdrant.
database = QdrantClient(
    url=metadata["url"],
    timeout=60
)

try:
    # Check the collection before making a Gemini request.
    if not database.collection_exists(metadata["collection"]):
        raise SystemExit("Collection not found. Run store.py first.")

    print("\nCreating the question embedding...")

    with genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=60000)
    ) as client:

        response = client.models.embed_content(
            model=metadata["model"],
            contents=question,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=metadata["dimensions"]
            )
        )

    if not response.embeddings:
        raise SystemExit("Gemini did not return an embedding.")

    question_vector = response.embeddings[0].values

    if not question_vector:
        raise SystemExit("Gemini returned an empty vector.")

    if len(question_vector) != metadata["dimensions"]:
        raise SystemExit("Question vector has an unexpected size.")

    print("Searching Qdrant...")

    # Find the closest stored document vectors.
    results = database.query_points(
        collection_name=metadata["collection"],
        query=question_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False
    ).points

finally:
    database.close()


# Collect and display the retrieved passages.
matches = []

for rank, result in enumerate(results, start=1):

    payload = result.payload or {}

    match = {
        "rank": rank,
        "id": result.id,
        "score": result.score,
        "source": payload["source"],
        "page": payload["page"],
        "text": payload["text"]
    }

    matches.append(match)

    print(f"\n--- Match {rank} ---")
    print(f"Source: {match['source']}")
    print(f"Page: {match['page']}")
    print(f"Similarity: {match['score']:.4f}")
    print(f"Text:\n{match['text']}")


# Save the question and passages for answer generation.
output_data = {
    "question": question,
    "matches": matches
}

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(output_data, file, indent=4, ensure_ascii=False)

if not matches:
    print("\nNo matching points were returned.")

print(f"\nSaved retrieved passages to: {output_path}")