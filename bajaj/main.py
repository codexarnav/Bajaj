import json
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


import os
import tempfile
import requests
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(path_or_url: str):
    """
    Loads and chunks a PDF from either a local file path or a remote HTTPS URL.
    Returns a list of LangChain Document chunks.
    """
    
    if path_or_url.lower().startswith("http"):
        print("🔽 Downloading PDF from URL...")
        response = requests.get(path_or_url)
        response.raise_for_status()

        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_pdf_path = tmp_file.name
    else:
        tmp_pdf_path = path_or_url  

    
    loader = PyMuPDFLoader(tmp_pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

    
    if path_or_url.lower().startswith("http"):
        os.remove(tmp_pdf_path)

    return chunks


def create_vector_store(chunks, embedding_model):
    return FAISS.from_documents(chunks, embedding_model)


def extract_structured_query(question, llm):
    messages = [
        SystemMessage(content="You are a smart assistant that extracts structured queries from user questions. "
                              "Return the **main topic or entity** the user is asking about."),
        HumanMessage(content=question)
    ]
    return llm.invoke(messages).content.strip()


def retrieve_and_refine_clause(question, structured_query, vector_store, embeddings, llm):
    query_embedding = embeddings.embed_query(structured_query)
    retrieved_docs = vector_store.similarity_search_by_vector(query_embedding, k=3)

    prompt_template = PromptTemplate.from_template("""
You're a policy assistant. Extract clause(s) from the following content that answer the user's query. They don't need to be verbatim, but should capture the essence of the clause.

User query: {query}
Content:
{chunk}

Extracted Clause:
""")

    for doc in retrieved_docs:
        prompt = prompt_template.format(query=question, chunk=doc.page_content)
        clause = llm.invoke(prompt).content.strip()
        if clause:  
            return clause
    return "No relevant clause found."


def run_batch_policy_qa(pdf_path, questions):
    print("Loading PDF and setting up")
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.1, api_key=os.getenv("GEMINI_API_KEY"))
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    chunks = load_and_chunk_pdf(pdf_path)
    vector_store = create_vector_store(chunks, embeddings)

    results = []
    for q in questions:
        print(f" Processing: {q}")
        structured_query = extract_structured_query(q, llm)
        print(f" Structured query: {structured_query}")
        clause = retrieve_and_refine_clause(q, structured_query, vector_store, embeddings, llm)
        print(f"Extracted Clause: {clause}\n")
        results.append(clause)

    return {"answers": results}


if __name__ == "__main__":
    pdf_path = input(" Enter path to your PDF document: ")

    print("\n Enter your questions (type 'done' when finished):")
    questions = []
    while True:
        q = input(f"Q{len(questions)+1}: ")
        if q.lower() == "done":
            break
        questions.append(q)

    output = run_batch_policy_qa(pdf_path, questions)

    print("\n Final JSON Output:")
    print(json.dumps(output, indent=2))
