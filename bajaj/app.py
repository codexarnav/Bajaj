from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

from main import (
    load_and_chunk_pdf,
    create_vector_store,
    extract_structured_query,
    retrieve_and_refine_clause
)

# ------------------------------
# Pydantic Schemas
# ------------------------------
class UserInput(BaseModel):
    documents: str = Field(description="Path or URL to the PDF document")
    questions: List[str] = Field(description="List of questions to ask about the document")

class Output(BaseModel):
    answers: List[str] = Field(description="Answers to the provided questions")

# ------------------------------
# FastAPI App
# ------------------------------
app = FastAPI(
    title="Policy Document QA",
    description="API to extract policy-related answers from uploaded or linked PDF documents.",
    version="1.0.0"
)

@app.post("/hackrx/run", response_model=Output)
async def process_document(user_input: UserInput):
    try:
        # Initialize models
        llm = ChatGoogleGenerativeAI(
            model='gemini-2.0-flash',
            temperature=0.1,
            api_key=os.getenv("GEMINI_API_KEY")
        )
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load + chunk the document
        chunks = load_and_chunk_pdf(user_input.documents)
        vector_store = create_vector_store(chunks, embeddings)

        # Process each question
        answers = []
        for question in user_input.questions:
            structured_query = extract_structured_query(question, llm)
            clause = retrieve_and_refine_clause(
                question=question,
                structured_query=structured_query,
                vector_store=vector_store,
                embeddings=embeddings,
                llm=llm
            )
            answers.append(clause)

        return Output(answers=answers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error: {str(e)}")
