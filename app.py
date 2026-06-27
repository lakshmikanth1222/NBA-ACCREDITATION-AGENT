import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ibm import ChatWatsonx
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_DB_DIR = "chroma_db"

def get_llm():
    """Initializes and returns the IBM Watsonx Chat Model (Granite Instruct)."""
    return ChatWatsonx(
        model_id="ibm/granite-8b-code-instruct", 
        url=os.getenv("IBM_URL", "https://us-south.ml.cloud.ibm.com"),
        project_id=os.getenv("IBM_PROJECT_ID"),
        api_key=os.getenv("IBM_API_KEY"),
        params={
            "max_new_tokens": 8192,
            "temperature": 0.1,
        }
    )

def get_retriever(k=4):
    """Initializes the Chroma vector store and returns the base (permanent) retriever."""
    if not os.path.exists(CHROMA_DB_DIR):
        raise FileNotFoundError(f"Chroma DB not found at '{CHROMA_DB_DIR}'. Please run ingest.py first.")
        
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    
    return db.as_retriever(search_type="mmr", search_kwargs={'k': k, 'fetch_k': 15})

def retrieve_dual_docs(query, base_retriever, temp_retriever=None):
    """Retrieves documents from both the permanent NBA database and the ephemeral user session database."""
    docs = base_retriever.invoke(query)
    if temp_retriever:
        try:
            temp_docs = temp_retriever.invoke(query)
            # Add a visual flag in the metadata so we know it came from the session
            for doc in temp_docs:
                doc.metadata['source_type'] = 'User Upload (Ephemeral)'
            docs.extend(temp_docs)
        except Exception as e:
            print(f"Error invoking temp_retriever: {e}")
    return docs

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ==========================================
# 1. Evidence Finder (QA Chain)
# ==========================================
def get_qa_chain(temp_retriever=None):
    llm = get_llm()
    base_retriever = get_retriever(k=4)
    
    system_prompt = (
        "You are an AI Accreditation Copilot specializing in NBA (National Board of Accreditation) guidelines.\n"
        "Answer the user's question based strictly on the retrieved context below.\n"
        "If you don't know the answer, say you cannot find it in the uploaded documents.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    
    return (
        RunnableParallel({"context": lambda x: retrieve_dual_docs(x["input"], base_retriever, temp_retriever), "input": lambda x: x["input"]})
        | RunnableParallel({
            "answer": {"context": lambda x: format_docs(x["context"]), "input": lambda x: x["input"]} | prompt | llm | StrOutputParser(),
            "context": lambda x: x["context"]
        })
    )

# ==========================================
# 2. AI SAR Generator Chain
# ==========================================
def get_sar_generation_chain(temp_retriever=None):
    llm = get_llm()
    base_retriever = get_retriever(k=6)
    
    system_prompt = (
        "You are an AI SAR (Self Assessment Report) Generator for NBA Accreditation.\n"
        "Based on the following context retrieved from the college's internal files and NBA manuals, "
        "draft the SAR section for the requested criterion.\n"
        "Format it professionally in markdown, highlighting key metrics, faculty data, or outcomes found in the context.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "Generate SAR section for: {input}")])
    
    return (
        RunnableParallel({"context": lambda x: retrieve_dual_docs(x["input"], base_retriever, temp_retriever), "input": lambda x: x["input"]})
        | RunnableParallel({
            "answer": {"context": lambda x: format_docs(x["context"]), "input": lambda x: x["input"]} | prompt | llm | StrOutputParser(),
            "context": lambda x: x["context"]
        })
    )

# ==========================================
# 3. Compliance & Gap Checker Chain
# ==========================================
def get_compliance_chain(temp_retriever=None):
    llm = get_llm()
    base_retriever = get_retriever(k=5)
    
    system_prompt = (
        "You are an NBA Compliance Checker.\n"
        "The user will provide a criterion or metric (e.g., 'Criterion 3').\n"
        "Read the retrieved documents. Compare what the NBA requires vs what the college has provided.\n"
        "List what is COMPLIANT (✓) and what is MISSING (❌). Provide a recommendation at the end.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "Check compliance for: {input}")])
    
    return (
        RunnableParallel({"context": lambda x: retrieve_dual_docs(x["input"], base_retriever, temp_retriever), "input": lambda x: x["input"]})
        | RunnableParallel({
            "answer": {"context": lambda x: format_docs(x["context"]), "input": lambda x: x["input"]} | prompt | llm | StrOutputParser(),
            "context": lambda x: x["context"]
        })
    )

# ==========================================
# 4. CO-PO Mapping Generator Chain
# ==========================================
def get_copo_mapping_chain():
    llm = get_llm()
    
    system_prompt = (
        "You are an expert in Outcome-Based Education (OBE).\n"
        "The user will provide a course syllabus. "
        "Your task is to:\n"
        "1. Identify or generate 4-5 Course Outcomes (COs).\n"
        "2. Map them to the standard 12 Program Outcomes (POs) and 2 Program Specific Outcomes (PSOs).\n"
        "3. Output the result as a Markdown table (Columns: CO, POs mapped, Justification).\n"
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "Syllabus:\n{input}")])
    
    return prompt | llm | StrOutputParser()
