import os
import tempfile
import streamlit as st

from torch import cuda, bfloat16
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM

# ---------- CONFIG ----------
EMBEDDING_MODEL_NAME = "intfloat/e5-small-v2"
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
PERSIST_DIR = "chroma_index"
COLLECTION_NAME = "pdf_collection"
# ----------------------------

# ----------- PDF & TEXT HANDLING -----------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return text_splitter.split_text(text)
# -------------------------------------------

# ----------- VECTOR STORE ------------------
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda" if cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_store = Chroma.from_texts(
        text_chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR
    )
    vector_store.persist()
    return vector_store.as_retriever()

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda" if cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return Chroma(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR
    ).as_retriever()
# -------------------------------------------

# ----------- TINYLLAMA WRAPPED LLM ----------
class WrappedTinyLlamaLLM(LLM):
    def __init__(self, generate_func):
        super().__init__()
        object.__setattr__(self, "generate_func", generate_func)

    def _call(self, prompt, stop=None):
        return self.generate_func(prompt)

    @property
    def _llm_type(self):
        return "custom-tinyllama"

def load_llm():
    model_id = LLM_MODEL_NAME
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )
    model_config = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
    )

    def prompt_wrapper(question, context):
        return f"### Context:\n{context}\n\n### Question:\n{question}\n\n### Answer:\n"

    def generate_func(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.split("Helpful Answer:")[-1].strip().split("###")[0].strip()

    def llm_call(prompt):
        return generate_func(prompt)

    return WrappedTinyLlamaLLM(llm_call)
# --------------------------------------------

# ----------- RAG Chain -----------------------
def get_conversational_chain(llm, retriever):
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", verbose=False)
# --------------------------------------------

# ------------- Streamlit UI ------------------
def main():
    st.set_page_config("Multi PDF Chatbot", page_icon="📚")
    st.header("📚 Multi-PDF Chat Bot 🤖")
    st.write("Upload PDF(s), embed once, and chat with documents intelligently!")

    retriever = None

    with st.sidebar:
        st.title("📁 PDF Upload")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("🔄 Processing documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    retriever = get_vector_store(text_chunks)
                    st.success("✅ Documents embedded and stored!")
            else:
                st.warning("Please upload at least one PDF.")

    # Load from persistent DB if exists
    if os.path.exists(PERSIST_DIR) and retriever is None:
        retriever = load_vector_store()

    if retriever:
        question = st.text_input("💬 Ask your question:")
        if st.button("Ask") and question.strip():
            llm = load_llm()
            qa_chain = get_conversational_chain(llm, retriever)
            with st.spinner("🔍 Thinking..."):
                response = qa_chain.run(question)
                st.markdown("### 🧠 Answer")
                st.write(response)
        elif question.strip() == "":
            st.info("Enter a question above to get started.")
    else:
        st.info("📥 Please upload and process PDFs to start chatting.")
# ---------------------------------------------

if __name__ == "__main__":
    main()
