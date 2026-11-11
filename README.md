# PDF RAG Bot

A Python application that answers questions about an indexed PDF using
Retrieval-Augmented Generation (RAG), Gemini, and Qdrant.

## Features

- Extracts text from PDFs and splits it into chunks.
- Generates Gemini embeddings for semantic search.
- Stores document vectors in Qdrant running in Docker.
- Retrieves relevant passages for a question.
- Generates answers with PDF page citations.
- Includes a Streamlit interface with expandable sources and answer downloads.

## Technologies

Python, Gemini API, Qdrant, Docker, Streamlit, and pypdf.

## Project Files

| File | Purpose |
|---|---|
| extract.py | Extract PDF text |
| chunk.py | Split extracted text into chunks |
| embed.py | Generate document embeddings |
| store.py | Upload embeddings and text to Qdrant |
| retrieve.py | Find passages relevant to a question |
| answer.py | Generate an answer from retrieved passages |
| main.py | Run the terminal application |
| app.py | Run the Streamlit web interface |
| requirements.txt | List Python dependencies |

## Setup on Windows

Install Python and Docker Desktop, then clone the repository:

```bat
git clone https://github.com/Prem-thangellapelly/pdf-rag-bot.git
cd pdf-rag-bot
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and enter your own Gemini API key:

```bat
copy .env.example .env
```

Start Docker Desktop, then create Qdrant:

```bat
docker run -d --name pdf-rag-qdrant -p 127.0.0.1:6333:6333 -v pdf_rag_qdrant_data:/qdrant/storage qdrant/qdrant
```

For later sessions, start the existing container:

```bat
docker start pdf-rag-qdrant
```

## Index a PDF

Place your PDF at the input location configured in `extract.py`.
Create that input folder if necessary.

Run the indexing steps in order:

```bat
python extract.py
python chunk.py
python embed.py
python store.py
```

Indexing generates local data files. The storage script stops if the
`pdf_chunks` collection already exists.

## Run the Web Interface

```bat
python -m streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Run the Terminal Interface

```bat
python main.py
```

Choose option 1 to ask a question about the indexed PDF.

## Configuration and Data

- Keep your real API key in `.env`; never commit it.
- `.env.example` contains configuration placeholders.
- PDFs, generated data, and local database files are excluded from Git.
- Each user setting up the project must supply a PDF and build an index.
- Gemini requests require internet access and use API quota.

## Limitations

- Each web question is processed independently; conversation history is
  not used to answer follow-up questions.
- Retrieved passages may not contain the answer.
- Generated answers and citations should be checked against the document.
- The app currently runs locally and requires the computer and Docker
  to stay running.