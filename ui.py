import os
import shutil
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from app import get_qa_chain, get_sar_generation_chain, get_compliance_chain, get_copo_mapping_chain

st.set_page_config(page_title="NBA AI Copilot", page_icon="🎓", layout="wide")

# --- Initialize Session State for Ephemeral DB ---
if "temp_retriever" not in st.session_state:
    st.session_state.temp_retriever = None
if "temp_doc_count" not in st.session_state:
    st.session_state.temp_doc_count = 0

# --- Sidebar Navigation ---
st.sidebar.title("NBA AI Copilot 🎓")
st.sidebar.markdown("Automate your NBA Accreditation Lifecycle.")

navigation = st.sidebar.radio(
    "Choose a Tool:",
    ["Evidence Finder", "AI SAR Generator", "Compliance Checker", "CO-PO Mapping Generator", "Readiness Dashboard"]
)

st.sidebar.divider()

# --- Ephemeral Document Management ---
st.sidebar.header("College Documents (Session Only)")
st.sidebar.write("Upload your college documents here. They will be processed in-memory and automatically deleted when you close this tab.")

if st.session_state.temp_doc_count > 0:
    st.sidebar.success(f"✅ {st.session_state.temp_doc_count} college documents currently active in memory.")
    if st.sidebar.button("🗑️ Clear Session Documents"):
        st.session_state.temp_retriever = None
        st.session_state.temp_doc_count = 0
        st.rerun()

uploaded_files = st.sidebar.file_uploader("Upload internal PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    if st.sidebar.button("Process Documents for this Session"):
        with st.spinner("Processing files securely in-memory..."):
            # Create a temporary directory to save files so PyPDFLoader can read them
            temp_dir = tempfile.mkdtemp()
            documents = []
            
            try:
                for uploaded_file in uploaded_files:
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    loader = PyPDFLoader(temp_path)
                    documents.extend(loader.load())
                
                # Split and Chunk
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_documents(documents)
                
                # Create In-Memory Chroma DB (no persist_directory)
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                in_memory_db = Chroma.from_documents(chunks, embeddings)
                
                # Store retriever in session state
                st.session_state.temp_retriever = in_memory_db.as_retriever(search_type="mmr", search_kwargs={'k': 4, 'fetch_k': 10})
                st.session_state.temp_doc_count = len(uploaded_files)
                
                st.sidebar.success("Documents processed successfully!")
                
            finally:
                # Clean up the physical files from the temporary directory immediately
                shutil.rmtree(temp_dir)
                st.rerun()

# ==========================================
# 1. Evidence Finder
# ==========================================
if navigation == "Evidence Finder":
    st.title("Evidence Finder 🔍")
    st.write("Search across the NBA rulebook and your uploaded session documents instantly.")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("Sources"):
                    st.markdown(msg["sources"])
                    
    if prompt := st.chat_input("E.g., Show all publications related to AI between 2022 and 2025."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Searching evidence..."):
                try:
                    chain = get_qa_chain(temp_retriever=st.session_state.temp_retriever)
                    response = chain.invoke({"input": prompt})
                    answer = response.get("answer", "No answer found.")
                    context = response.get("context", [])
                    
                    st.markdown(answer)
                    
                    source_md = ""
                    if context:
                        source_md = "**Citations:**\n\n"
                        for i, doc in enumerate(context, 1):
                            src = os.path.basename(doc.metadata.get('source', 'Unknown'))
                            pg = doc.metadata.get('page', 'Unknown')
                            source_type = doc.metadata.get('source_type', 'NBA Permanent Rulebook')
                            source_md += f"{i}. **{src}** (Page {pg}) - *{source_type}*\n"
                        with st.expander("Sources"):
                            st.markdown(source_md)
                            
                    st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": source_md})
                except Exception as e:
                    st.error(f"Error: {e}")

# ==========================================
# 2. AI SAR Generator
# ==========================================
elif navigation == "AI SAR Generator":
    st.title("AI SAR Generator 📝")
    st.write("Automatically generate Self Assessment Report sections based on your uploaded college files.")
    
    criterion = st.selectbox(
        "Select Criterion to Generate:",
        ["Criterion 1: Vision, Mission and Program Educational Objectives",
         "Criterion 2: Program Curriculum and Teaching-Learning Processes",
         "Criterion 3: Course Outcomes and Program Outcomes",
         "Criterion 4: Students' Performance",
         "Criterion 5: Faculty Information and Contributions"]
    )
    
    if st.button("Generate SAR Section", type="primary"):
        with st.spinner(f"Analyzing documents and generating {criterion}..."):
            try:
                chain = get_sar_generation_chain(temp_retriever=st.session_state.temp_retriever)
                response = chain.invoke({"input": criterion})
                answer = response.get("answer", "")
                st.markdown(answer)
                st.download_button("Download Markdown", data=answer, file_name=f"{criterion[:11]}.md", mime="text/plain")
            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# 3. Compliance Checker
# ==========================================
elif navigation == "Compliance Checker":
    st.title("Compliance & Gap Checker 🛡️")
    st.write("Compare your uploaded college documents against the official NBA guidelines to find gaps.")
    
    query = st.text_input("What would you like to check?", "Are we compliant with Criterion 3?")
    if st.button("Run Compliance Check"):
        if not st.session_state.temp_retriever:
            st.warning("You haven't uploaded any college documents for this session! The AI will only search the NBA rules.")
            
        with st.spinner("Running compliance audit..."):
            try:
                chain = get_compliance_chain(temp_retriever=st.session_state.temp_retriever)
                response = chain.invoke({"input": query})
                st.markdown(response.get("answer", ""))
            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# 4. CO-PO Mapping Generator
# ==========================================
elif navigation == "CO-PO Mapping Generator":
    st.title("CO-PO Mapping Generator 📊")
    st.write("Paste your course syllabus, and AI will generate Course Outcomes and map them to Program Outcomes.")
    
    syllabus = st.text_area("Paste Course Syllabus here:", height=250)
    
    if st.button("Generate Mappings", type="primary"):
        if not syllabus:
            st.warning("Please provide a syllabus.")
        else:
            with st.spinner("Analyzing syllabus and generating mappings..."):
                try:
                    chain = get_copo_mapping_chain()
                    response = chain.invoke({"input": syllabus})
                    st.markdown(response)
                except Exception as e:
                    st.error(f"Error: {e}")

# ==========================================
# 5. Readiness Dashboard
# ==========================================
elif navigation == "Readiness Dashboard":
    st.title("Accreditation Readiness Dashboard 📈")
    st.write("Overview of your NBA Accreditation status.")
    
    st.subheader("Criterion Readiness")
    col1, col2, col3 = st.columns(3)
    col1.metric("Criterion 1 (Vision/Mission)", "92%", "+2%")
    col2.metric("Criterion 2 (Teaching-Learning)", "81%", "-5%")
    col3.metric("Criterion 3 (CO-PO)", "75%", "Needs Improvement")
    
    col4, col5, col6 = st.columns(3)
    col4.metric("Criterion 4 (Students)", "96%", "Excellent")
    col5.metric("Criterion 5 (Faculty)", "60%", "Critical Action Required")
    col6.metric("Overall Readiness", "81%", "Not Ready")
    
    st.divider()
    st.subheader("Actionable Recommendations (Evidence Recommendation Engine)")
    st.warning("To improve Criterion 3 (CO-PO): Upload missing Lab CO Attainment sheets.")
    st.error("To improve Criterion 5 (Faculty): Upload Industry Consultancies and Patent details.")
    st.success("Criterion 4 (Students) is well documented. No action needed.")
