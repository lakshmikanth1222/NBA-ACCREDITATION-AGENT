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
            "temperature": 0.0,
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
        "You are an expert Accreditation Consultant for the National Board of Accreditation (NBA) in India, which accredits engineering colleges.\n"
        "CRITICAL WARNING: 'NBA' here means Engineering Accreditation, NOT the National Basketball Association. DO NOT write about basketball!\n"
        "You MUST base your entire report ONLY on the provided Context. Extract specific institution names (e.g. Sir C.R. Reddy College), department names, exact statistics, and hard facts.\n"
        "DO NOT output generic filler text. If data is missing, explicitly state 'Data not available in provided evidence'.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt), 
        ("human", "Draft a highly professional, visually impactful NBA Accreditation SAR section for: {input}\n\n"
                  "You MUST use this EXACT markdown structure to ensure it is visually impactful:\n\n"
                  "## 🏫 Criterion Report: {input}\n\n"
                  "### 📊 1. Core Data & Extracted Metrics\n"
                  "- **Metric 1:** (Extract a specific hard fact/number from the context)\n"
                  "- **Metric 2:** (Extract another specific hard fact/number)\n"
                  "- **Metric 3:** (Extract another specific hard fact/number)\n\n"
                  "### 📝 2. Detailed Academic Analysis\n"
                  "(Write 1-2 paragraphs of detailed analysis based on the context. Use standard NBA sub-headings like 1.1, 1.2. Use **bold text** to highlight key college names or achievements.)\n\n"
                  "### ✅ 3. Compliance Summary\n"
                  "> **Overall Status:** (State if compliant based on evidence)\n"
                  "> **Justification:** (Brief 1-sentence justification)")
    ])
    
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
        "The user will provide a course syllabus. Your task is to:\n"
        "1. Generate exactly 4 Course Outcomes (COs).\n"
        "2. Map them to the 12 Program Outcomes (POs).\n"
        "3. Output the result EXACTLY as a Markdown table.\n"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt), 
        ("human", "Here is the course syllabus:\n\n{input}\n\nBased on this syllabus, generate the CO-PO mapping. You MUST output a Markdown table with EXACTLY these three columns:\n\n| Course Outcome (CO) | Mapped POs | Justification |\n|---|---|---|")
    ])
    
    return prompt | llm | StrOutputParser()

# ==========================================
# 5. Dashboard Evaluation Chain
# ==========================================
def get_dashboard_evaluation_chain(temp_retriever=None):
    llm = get_llm()
    base_retriever = get_retriever(k=4)
    
    system_prompt = (
        "You are an AI Accreditation Copilot. Evaluate the uploaded college documents against the NBA rules.\n"
        "Generate a readiness score from 0 to 100 for each of the 5 criteria, based on how much evidence is provided.\n"
        "IMPORTANT: You MUST output ONLY valid JSON. Do not write any markdown, markdown blocks, or conversational text. Output exactly this format:\n"
        "{{\n"
        "  \"Criterion 1\": {{\"score\": 90, \"feedback\": \"Good\"}},\n"
        "  \"Criterion 2\": {{\"score\": 80, \"feedback\": \"Missing XYZ\"}},\n"
        "  \"Criterion 3\": {{\"score\": 75, \"feedback\": \"...\"}},\n"
        "  \"Criterion 4\": {{\"score\": 95, \"feedback\": \"...\"}},\n"
        "  \"Criterion 5\": {{\"score\": 60, \"feedback\": \"...\"}},\n"
        "  \"Overall\": 80\n"
        "}}\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "Evaluate readiness and return ONLY JSON for: {input}")])
    
    return (
        RunnableParallel({"context": lambda x: retrieve_dual_docs(x["input"], base_retriever, temp_retriever), "input": lambda x: x["input"]})
        | prompt 
        | llm 
        | StrOutputParser()
    )
