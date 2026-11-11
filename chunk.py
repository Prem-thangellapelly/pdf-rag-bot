from pathlib import Path
import json


# Locate the project folder.
project_folder = Path(__file__).resolve().parent

# Define the input and output files.
input_path = project_folder / "data" / "extracted_pages.json"
output_path = project_folder / "data" / "chunks.json"

# These sizes are measured in characters.
chunk_size = 1000
chunk_overlap = 200


# Check whether extraction has been completed.
if not input_path.exists():
    raise SystemExit("Extracted text not found. Run extract.py first.")


# Load the extracted pages into a Python list.
with open(input_path, "r", encoding="utf-8") as file:
    extracted_pages = json.load(file)


# Store all the chunks here.
chunks = []


# Process one page at a time.
for page in extracted_pages:

    text = page["text"]
    start = 0

    while start < len(text):

        # Select up to 1,000 characters.
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]

        # Keep the text and its original page information.
        chunk_data = {
            "id": len(chunks),
            "source": page["source"],
            "page": page["page"],
            "text": chunk_text
        }

        chunks.append(chunk_data)

        # Stop after reaching the end of this page.
        if end == len(text):
            break

        # Keep 200 characters from the previous chunk.
        start = end - chunk_overlap


# Save the chunks for the embedding step.
with open(output_path, "w", encoding="utf-8") as file:
    json.dump(chunks, file, indent=4, ensure_ascii=False)


print(f"Pages processed: {len(extracted_pages)}")
print(f"Chunks created: {len(chunks)}")
print(f"Saved chunks to: {output_path}")