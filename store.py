from pathlib import Path
from qdrant_client import QdrantClient, models
import json


# Locate the project folder.
project_folder = Path(__file__).resolve().parent

# Define input and metadata files.
input_path = project_folder / "data" / "embeddings.json"
metadata_path = project_folder / "data" / "index_meta.json"

# Connect to the Qdrant server running in Docker.
qdrant_url = "http://localhost:6333"
collection_name = "pdf_chunks"


# Check whether embeddings exist.
if not input_path.exists():
    raise SystemExit("Embeddings not found. Run embed.py first.")


# Load the saved embeddings.
with open(input_path, "r", encoding="utf-8") as file:
    embedding_data = json.load(file)

records = embedding_data["records"]
dimensions = embedding_data["dimensions"]

if not records:
    raise SystemExit("The embeddings file contains no records.")


# Prepare Qdrant points.
points = []

for record in records:

    if len(record["vector"]) != dimensions:
        raise SystemExit("A vector has an unexpected size.")

    point = models.PointStruct(
        id=record["id"],
        vector=record["vector"],
        payload={
            "text": record["text"],
            "source": record["source"],
            "page": record["page"]
        }
    )

    points.append(point)


# Create the connection to Docker Qdrant.
client = QdrantClient(
    url=qdrant_url,
    timeout=60
)

try:
    # Confirm that Qdrant is reachable.
    client.get_collections()
    print(f"Connected to Qdrant: {qdrant_url}")

    # Stop if this collection already exists.
    if client.collection_exists(collection_name):
        raise SystemExit(
            "The pdf_chunks collection already exists. "
            "If you stored this PDF successfully before, "
            "continue to retrieval."
        )

    # Create a collection matching the embedding dimensions.
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=dimensions,
            distance=models.Distance.COSINE
        )
    )

    # Upload the vectors and their text.
    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True
    )

    # Verify the number of stored points.
    stored_count = client.count(
        collection_name=collection_name,
        exact=True
    ).count

    if stored_count != len(records):
        raise SystemExit(
            f"Expected {len(records)} points, "
            f"but found {stored_count}. Storage needs checking."
        )

finally:
    # Close Python's connection. The Docker server keeps running.
    client.close()


# Save settings for the retrieval script.
index_metadata = {
    "provider": embedding_data["provider"],
    "model": embedding_data["model"],
    "dimensions": dimensions,
    "collection": collection_name,
    "url": qdrant_url
}

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(index_metadata, file, indent=4)


print(f"Collection: {collection_name}")
print(f"Vector dimensions: {dimensions}")
print(f"Chunks stored: {stored_count}")
print(f"Metadata saved to: {metadata_path}")
print("Dashboard: http://localhost:6333/dashboard")