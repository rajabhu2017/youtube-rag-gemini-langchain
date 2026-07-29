# youtube-rag-gemini-langchain
## YouTube RAG using LangChain + Google Gemini + FAISS

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about any YouTube video using its transcript.

The application automatically downloads the transcript, converts it into vector embeddings using **Google Gemini Embeddings**, stores them in a **FAISS Vector Database**, and answers user queries using **Gemini 2.5 Flash**.

To optimize performance and reduce API usage, a separate FAISS database is created for each YouTube video and reused in future sessions.

---

## 🚀 Features

- Ask questions about any YouTube video
- Automatic transcript retrieval
- Text chunking using Recursive Character Text Splitter
- Google Gemini Embeddings
- FAISS Vector Database
- Retrieval-Augmented Generation (RAG)
- Automatic local caching of vector databases
- Supports multiple YouTube videos
- Built with the latest LangChain APIs

---

## 🛠 Tech Stack

- Python
- LangChain
- Google Gemini API
- FAISS
- YouTube Transcript API
- dotenv

---

## 📂 Project Structure

```
YouTube-RAG
│
├── youtube_rag.py
├── .env
├── requirements.txt
├── README.md
│
└── youtube_faiss_db
    ├── Gfr50f6ZBvo
    │     ├── index.faiss
    │     └── index.pkl
    │
    ├── AnotherVideoID
    │     ├── index.faiss
    │     └── index.pkl
    │
    └── ...
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/youtube-rag.git
```

Move into the project directory

```bash
cd youtube-rag
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```text
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
```

You can generate an API key from:

https://aistudio.google.com/app/apikey

---

## ▶️ Run the Application

```bash
python youtube_rag.py
```

Example:

```
Enter YouTube Video ID:

Gfr50f6ZBvo
```

Then ask questions such as:

```
Summarize the video.

Who is discussed in the video?

What are the key takeaways?

Explain the main concepts.
```

---

## 🧠 How It Works

### First Run

```
YouTube Video
        │
        ▼
Download Transcript
        │
        ▼
Split into Chunks
        │
        ▼
Generate Embeddings
        │
        ▼
Create FAISS Database
        │
        ▼
Save Locally
```

---

### Future Runs

```
Video ID
      │
      ▼
Load Existing FAISS Database
      │
      ▼
Retrieve Relevant Chunks
      │
      ▼
Gemini 2.5 Flash
      │
      ▼
Answer User Query
```

Since embeddings are generated only once, subsequent executions are significantly faster and consume fewer API requests.

---

## 📌 Example Workflow

```
Enter Video ID

↓

Transcript Download

↓

FAISS Database Created

↓

Database Saved

↓

Ask Unlimited Questions
```

On future executions:

```
Enter Same Video ID

↓

Load Existing Database

↓

Ask Questions Instantly
```

---

## 📈 Advantages

- Faster execution after the first run
- Lower Gemini API usage
- Scalable architecture
- Separate vector database for every video
- Easy to extend to PDFs, websites, and other document sources

---

## 📚 Future Improvements

- Streamlit Web UI
- Chat History / Memory
- Multiple video retrieval
- Hybrid Search (BM25 + FAISS)
- ChromaDB support
- Source citations with timestamps
- Conversation memory
- Multi-modal RAG
