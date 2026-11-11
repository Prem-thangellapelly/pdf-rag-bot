from pathlib import Path
from pypdf import PdfReader
import json


# Find the folder containing this Python file.
project_folder = Path(__file__).resolve().parent

# Define where the PDF is located.
pdf_path = project_folder / "data" / "document.pdf"
# Create a folder for the output files.
output_folder = project_folder / "data"
output_folder.mkdir(exist_ok=True)

# Define the output filename.
output_path = output_folder / "extracted_pages.json"


# Check whether the PDF exists.
if not pdf_path.exists():
    raise SystemExit("PDF not found. Put document.pdf inside the input folder.")


# Open the PDF.
reader = PdfReader(pdf_path)

# Store the extracted page information.
extracted_pages = []


# Read each page separately.
for page_number, page in enumerate(reader.pages, start=1):

    text = (page.extract_text() or "").strip()

    if text:
        page_data = {
            "source": pdf_path.name,
            "page": page_number,
            "text": text
        }

        extracted_pages.append(page_data)

        print(f"Extracted page {page_number}")

    else:
        print(f"Page {page_number}: no text found")


# Stop if the PDF contains no extractable text.
if not extracted_pages:
    raise SystemExit("No text extracted. This PDF may require OCR.")


# Save the extracted text and page information.
with open(output_path, "w", encoding="utf-8") as file:
    json.dump(
        extracted_pages,
        file,
        indent=4,
        ensure_ascii=False
    )

print(f"\nSaved extracted text to: {output_path}")