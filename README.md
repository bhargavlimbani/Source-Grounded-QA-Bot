# Source-Grounded Q&A Bot

A Streamlit application that answers questions from uploaded documents. It retrieves relevant passages, asks Gemini to answer from those passages only, and displays source citations. When the available evidence is insufficient, the app returns **“Not found in the document.”**

## Features

- Upload multiple PDF, DOCX, or TXT documents.
- Extract text and split it into sentence-based passages with document and page/section details.
- Retrieve relevant passages using TF-IDF and cosine similarity.
- Compare two prompt versions: V1 uses one answer-generation call; V2 adds few-shot examples and a second verification call.
- Reject answers with missing or unknown citations and decline questions below the retrieval relevance threshold.
- Review the retrieved evidence, session question history, and export history as CSV.
- Enter up to 10 evaluation cases and record faithfulness results in the app.

## Requirements

- Python 3.10 or newer
- A Google Gemini API key
- Packages listed in [`requirements.txt`](requirements.txt)

## Run locally

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:GEMINI_API_KEY = "your_api_key"
streamlit run app.py
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export GEMINI_API_KEY="your_api_key"
streamlit run app.py
```

Open the local URL printed by Streamlit. Set `GEMINI_MODEL` to override the default model (`gemini-3.8-flash`), if needed.

## Use the app

1. Upload one or more PDF, DOCX, or TXT files in the sidebar.
2. Choose V1 or V2, enter a question, and select **Ask Question**.
3. Review the answer, citations, retrieved passages, and relevance scores.
4. Use **Prompt Comparison** to submit the same question to both prompt versions.
5. Fill in the evaluation table to record results, and use the history section to review or download session questions.

The evaluation table starts blank. [`evaluation/test_questions.csv`](evaluation/test_questions.csv) is a sample template; replace its placeholder questions and expected labels with your own cases. Evaluation scores are based on the values entered by the user; the app does not run an automatic benchmark from this CSV.

## How answers are grounded

The app retrieves up to five passages using TF-IDF cosine similarity. If the best score is below `0.08`, it returns the not-found response without calling Gemini. For V1, one Gemini call generates an answer. For V2, a second call reviews the draft against retrieved context. The application also checks that citations refer to retrieved source IDs and that a supported answer includes citations.

These checks help keep answers tied to the uploaded material, but they do not guarantee that every generated statement or citation is correct. Review the cited passages for important decisions.

## Project layout

```text
.
├── app.py                         Streamlit application
├── requirements.txt               Python dependencies
├── PROMPT_HISTORY.md              Prompt design notes and evaluation record
├── documents/
│   └── PUT_YOUR_PDFS_HERE.txt     Reminder; upload documents in the app
└── evaluation/
    └── test_questions.csv         Editable example evaluation template
```

## Configuration and security

- Provide `GEMINI_API_KEY` through your shell environment. Do not put API keys in source code or commit them to Git.
- `GEMINI_MODEL` is optional and defaults to `gemini-3.8-flash`.
- Uploaded files are processed in the app session; this repository does not include source documents.
- The local Python virtual environment and common secret files are excluded by `.gitignore`.

## Prompt history

See [`PROMPT_HISTORY.md`](PROMPT_HISTORY.md) for the prompt versions, prompting techniques, guardrails, and space to record real evaluation results.
