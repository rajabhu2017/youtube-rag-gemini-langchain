from dotenv import load_dotenv
load_dotenv()

import os

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough,
)

from langchain_core.output_parsers import StrOutputParser


# ----------------------------------------------------
# Enter YouTube Video ID
# ----------------------------------------------------

video_id = input("Enter YouTube Video ID : ").strip()


# Example
# https://www.youtube.com/watch?v=Gfr50f6ZBvo
# Video ID = Gfr50f6ZBvo


# ----------------------------------------------------
# Embedding Model
# ----------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# ----------------------------------------------------
# FAISS Folder
# ----------------------------------------------------

BASE_FOLDER = "youtube_faiss_db"

os.makedirs(BASE_FOLDER, exist_ok=True)

DB_PATH = os.path.join(
    BASE_FOLDER,
    video_id
)


# ----------------------------------------------------
# Load Existing DB
# ----------------------------------------------------

if os.path.exists(DB_PATH):

    print("\nLoading existing FAISS database...\n")

    vectorstore = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

else:

    print("\nCreating FAISS database...\n")

    try:

        transcript = YouTubeTranscriptApi().fetch(
            video_id,
            languages=["en"]
        )

        transcript_text = " ".join(
            chunk.text
            for chunk in transcript
        )

    except TranscriptsDisabled:

        print("Transcript unavailable.")
        exit()

    except Exception as e:

        print(e)
        exit()


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    documents = splitter.create_documents(
        [transcript_text]
    )


    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )


    vectorstore.save_local(DB_PATH)

    print("\nDatabase Saved Successfully.\n")


# ----------------------------------------------------
# Retriever
# ----------------------------------------------------

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":4}
)


# ----------------------------------------------------
# Gemini
# ----------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
)


# ----------------------------------------------------
# Prompt
# ----------------------------------------------------

prompt = PromptTemplate.from_template(
"""
You are an expert assistant.

Answer ONLY from the transcript.

If the answer is not available in the transcript,
reply only:

I don't know.

Context:
{context}

Question:
{question}
"""
)


# ----------------------------------------------------
# Helper Function
# ----------------------------------------------------

def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# ----------------------------------------------------
# RAG Chain
# ----------------------------------------------------

parallel_chain = RunnableParallel(
{
    "context":
        retriever
        | RunnableLambda(format_docs),

    "question":
        RunnablePassthrough(),
}
)

parser = StrOutputParser()

chain = (
    parallel_chain
    | prompt
    | llm
    | parser
)


# ----------------------------------------------------
# Chat Loop
# ----------------------------------------------------

while True:

    question = input("\nAsk Question (exit to quit): ")

    if question.lower() == "exit":
        break

    answer = chain.invoke(question)

    print("\nAnswer:\n")

    print(answer)