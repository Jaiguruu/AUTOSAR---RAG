"""Streamlit app for AUTOSAR RAG with document retrieval and chat interface."""

from typing import List
import os
import re
import debugpy
import streamlit as st
from phi.document import Document
from phi.utils.log import logger
from PIL import Image
from assistant.assistant import get_rag_assistant
from posthdl.puml2img import generate_plantuml_image
from autosar_rag.autosar_loader import AutosarLoader
from autosar_rag.autosar_splitter import AutosarSplitter
from autosar_rag.structured_extractor import StructuredExtractor
from autosar_rag.validator import ConsistencyValidator
import json
import pandas as pd
from phi.embedder.openai import OpenAIEmbedder

# app.py
with open('assets/autosar_basic.puml', 'r', encoding='utf-8') as file:
    autosar_basic_puml = file.read()

def extract_plantuml_code(response):
    """从响应中提取 PlantUML 代码"""
    pattern = r'@startuml[\s\S]*?@enduml'
    match = re.search(pattern, response)
    if match:
        return match.group(0)
    return None

def restart_assistant():
    """Restart assistant."""
    st.session_state["rag_assistant"] = None
    st.session_state["rag_assistant_run_id"] = None
    if "url_scrape_key" in st.session_state:
        st.session_state["url_scrape_key"] += 1
    if "file_uploader_key" in st.session_state:
        st.session_state["file_uploader_key"] += 1
    st.rerun()

def configure_embedding_backend(embeddings_model):
    """Create an OpenAI embedder when the selected embedding model is OpenAI-based."""
    if embeddings_model.startswith("text-embedding-") or "openai" in embeddings_model.lower():
        st.session_state["openai_embedder"] = OpenAIEmbedder(model=embeddings_model)
    else:
        st.session_state.pop("openai_embedder", None)


def update_llm_config(llm_model, embeddings_model):
    """Check llm config."""
    # Set assistant_type in session state
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = llm_model
    # Restart the assistant if assistant_type has changed
    elif st.session_state["llm_model"] != llm_model:
        st.session_state["llm_model"] = llm_model
        restart_assistant()

    if "embeddings_model" not in st.session_state:
        st.session_state["embeddings_model"] = embeddings_model
    # Restart the assistant if assistant_type has changed
    elif st.session_state["embeddings_model"] != embeddings_model:
        st.session_state["embeddings_model"] = embeddings_model
        st.session_state["embeddings_model_updated"] = True
        restart_assistant()

    configure_embedding_backend(embeddings_model)

def get_or_create_assistant(llm_model, embeddings_model, custom_prompt, openrouter_api_key=None):
    """Get or create assistant."""
    if ("rag_assistant" not in st.session_state or
        st.session_state["rag_assistant"] is None or
        st.session_state.get("custom_prompt") != custom_prompt):

        logger.info("---*--- Creating %s Assistant ---*---", llm_model)
        rag_assistant = get_rag_assistant(
            llm_model=llm_model,
            embeddings_model=embeddings_model,
            instructions=custom_prompt,
            openrouter_api_key=openrouter_api_key
        )
        st.session_state["rag_assistant"] = rag_assistant
        st.session_state["custom_prompt"] = custom_prompt  # Store the custom prompt in session state
    else:
        rag_assistant = st.session_state["rag_assistant"]

    try:
        st.session_state["rag_assistant_run_id"] = rag_assistant.create_run()
    except (ConnectionError, TimeoutError) as e:
        st.warning(f"Could not create assistant: {str(e)}. Is the database running?")
        return None

    return rag_assistant

def load_assistant_chat_history(rag_assistant):
    """Load chat history from the assistant's memory."""
    if not rag_assistant:
        st.session_state["messages"] = [{
            "role": "assistant", 
            "content": "Please verify local Ollama and database status to start chatting."
        }]
        return

    assistant_chat_history = rag_assistant.memory.get_chat_history()
    if len(assistant_chat_history) > 0:
        logger.debug("Loading chat history")
        st.session_state["messages"] = assistant_chat_history
    else:
        logger.debug("No chat history found")
        st.session_state["messages"] = [{
            "role": "assistant", 
            "content": "Upload a doc or ask me questions about AUTOSAR directly..."
        }]

def load_knowledge_base(rag_assistant):
    """Load knowledge base."""
    if not rag_assistant or not rag_assistant.knowledge_base:
        return

    # Add PDFs to knowledge base
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 100

    uploaded_file = st.sidebar.file_uploader(
        "Add a PDF :page_facing_up:", type="pdf", key=st.session_state["file_uploader_key"]
    )
    if uploaded_file is not None:
        process_uploaded_file(rag_assistant, uploaded_file)

    if rag_assistant.knowledge_base.vector_db:
        if st.sidebar.button("Clear Knowledge Base", help="Clear all documents from the knowledge base."):
            rag_assistant.knowledge_base.vector_db.drop()
            st.sidebar.success("Knowledge base cleared")

def process_uploaded_file(rag_assistant, uploaded_file):
    """Process the uploaded PDF file."""
    alert = st.sidebar.info("Processing PDF...", icon="🧠")
    rag_name = uploaded_file.name.split(".")[0]
    if f"{rag_name}_uploaded" not in st.session_state:
        # chunk_size and separators can be set here
        logger.debug(uploaded_file)
        loader = AutosarLoader(uploaded_file)
        splitter = AutosarSplitter(chunk_size=300, chunk_overlap=20)
        raw_documents: List[Document] = loader.load()
        rag_documents: List[Document] = splitter.split_documents(raw_documents)
        phi_rag_documents: List[Document] = []
        # there are some difference in docment defination between langchain Document and phidata Document
        # so, convert them here for next operation
        for doc in rag_documents:
            phi_rag_documents.append(Document(content=doc.page_content,name=doc.metadata["source"], meta_data=doc.metadata))
        if rag_documents:
            print("Documents to index:", len(phi_rag_documents))

            for i, doc in enumerate(phi_rag_documents[:5]):
                content = getattr(doc, "content", None)
                embedding = getattr(doc, "embedding", None)

                print(
                    f"Document {i}:",
                    "content_length =", len(content or ""),
                    "embedding_length =",
                    len(embedding) if embedding is not None else None,
                )

                if not content or not content.strip():
                    print("WARNING: Empty document content")    
            rag_assistant.knowledge_base.load_documents(phi_rag_documents, upsert=True)
        else:
            st.sidebar.error("Could not read PDF")
        st.session_state[f"{rag_name}_uploaded"] = True
    alert.empty()

def initialize_debugger(enable_debug: bool = False):
    """Initialize debugpy for remote debugging if enabled."""
    if enable_debug and "debugpy_initialized" not in st.session_state:
        st.session_state.debugpy_initialized = True
        debugpy.listen(("localhost", 5678))
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()

def set_page_config():
    """Set page config."""
    st.image("assets/robot_autosar.jpeg", width=700)  # Adjust the width as needed
    st.markdown("## :oncoming_automobile: TATA Techpulse AUTOSAR Assistant")
    st.markdown("Welcome to the TATA Techpulse AUTOSAR Assistant. Fast local inference powered by **Qwen2.5-Coder:1.5B** via Ollama.")
    st.markdown("---")

def update_session_content(session_messages, rag_assistant):
    """Update session content."""
    # Prompt for user input using a chat input box from Streamlit
    if prompt := st.chat_input():
        session_messages.append({"role": "user", "content": prompt})

    # Display existing chat messages
    for message in session_messages:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is from a user, generate a new response
    last_message = session_messages[-1]
    if last_message.get("role") == "user":
        if not rag_assistant:
            with st.chat_message("assistant"):
                st.error("Assistant is not initialized. Please ensure Ollama and database services are running.")
            return

        question = last_message["content"]
        if "/bd" in question:
            question = question.replace("/bd", "").strip()
            question += " \n Generate corresponding plantUML code(marked with startuml and enduml) of block diagram based on the following plantUML code of high level AUTOSAR architecture block digram: \n"
            question += autosar_basic_puml
            with st.chat_message("assistant"):
                response = ""
                resp_container = st.empty()
                for delta in rag_assistant.run(question):
                    response += delta  # type: ignore
                    resp_container.markdown(response)
                session_messages.append({"role": "assistant", "content": response})

                # Check if the response contains PlantUML code and export it
                plantuml_code = extract_plantuml_code(response)
                if plantuml_code:
                    with open("generated_diagram.puml", "w", encoding="utf-8") as puml_file:
                        puml_file.write(plantuml_code)

                # Add the generated plantUML image and display it in a new popup window
                image_path = "output.png"
                generate_plantuml_image("generated_diagram.puml", image_path)
                if os.path.exists(image_path):
                    image = Image.open(image_path)
                    st.session_state["messages"].append({"role": "assistant", "content_type": "image", "content": image})
                    st.image(image, caption="Generated AUTOSAR PlantUML Diagram")

        else:
            with st.chat_message("assistant"):
                response = ""
                resp_container = st.empty()
                for delta in rag_assistant.run(question):
                    response += delta  # type: ignore
                    resp_container.markdown(response)
                session_messages.append({"role": "assistant", "content": response})

def load_assistant_storage(rag_assistant, llm_model, embeddings_model):
    """Load assistant storage."""
    if not rag_assistant or not rag_assistant.storage:
        return
    rag_assistant_run_ids: List[str] = rag_assistant.storage.get_all_run_ids()
    if not rag_assistant_run_ids:
        return
    new_rag_assistant_run_id = st.sidebar.selectbox("Run ID", options=rag_assistant_run_ids)
    if st.session_state["rag_assistant_run_id"] != new_rag_assistant_run_id:
        logger.info("---*--- Loading %s run: %s ---*---", llm_model, new_rag_assistant_run_id)
        st.session_state["rag_assistant"] = get_rag_assistant(
            llm_model=llm_model, embeddings_model=embeddings_model, run_id=new_rag_assistant_run_id
        )
        st.rerun()

def run_structured_analysis(rag_assistant, llm_model="qwen2.5-coder:1.5b", openrouter_api_key=None):
    """Run structured extraction, validation, and export workflow."""
    extractor = StructuredExtractor(llm_model=llm_model, openrouter_api_key=openrouter_api_key)
    validator = ConsistencyValidator()

    all_documents = []
    if rag_assistant and rag_assistant.knowledge_base:
        # Search for relevant chunks to extract from
        all_documents = rag_assistant.knowledge_base.search(
            "AUTOSAR component architecture ports interfaces"
        )
        all_documents = all_documents[:10]
    if not all_documents:
        st.warning("No documents loaded. Please upload a PDF first.")
        return

    extractions = []
    for doc in all_documents:
        extractions.append(extractor.extract_from_text(doc.content))

    structured_model = extractor.merge_extractions(extractions)

    if structured_model["components"]:
        st.markdown("### Extracted Component Inventory")
        df = pd.DataFrame([{"Component": c["name"], "Type": c.get("type", "N/A"), "Ports Count": len(c.get("ports", []))} for c in structured_model["components"]])
        st.dataframe(df)

        # Export
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Download Components JSON", json.dumps(structured_model, indent=4), file_name="architecture.json")
        with col2:
            st.download_button("Download Components CSV", df.to_csv(index=False), file_name="components_inventory.csv")

    # Consistency Check
    findings = validator.validate_model(structured_model)
    st.markdown("### Consistency Report")
    if findings:
        for f in findings:
            if f["severity"] == "Error":
                st.error(f"[{f['component']}] {f['issue']}")
            else:
                st.warning(f"[{f['component']}] {f['issue']}")
    else:
        st.success("No inconsistencies detected in the current model.")

def run_revision_comparison():
    """Compare two structured models from uploaded revisions."""
    st.markdown("### Document Revision Comparison")
    col1, col2 = st.columns(2)
    with col1:
        old_file = st.file_uploader("Upload Old Revision JSON", type="json")
    with col2:
        new_file = st.file_uploader("Upload New Revision JSON", type="json")

    if old_file and new_file:
        try:
            old_model = json.load(old_file)
            new_model = json.load(new_file)
            validator = ConsistencyValidator()
            diff = validator.compare_revisions(old_model, new_model)

            if diff["added_components"]:
                st.markdown("**Newly Added Components:**")
                st.success(", ".join(diff["added_components"]))
            if diff["removed_components"]:
                st.markdown("**Removed Components:**")
                st.error(", ".join(diff["removed_components"]))
            if diff["modified_components"]:
                st.markdown("**Modified Components:**")
                st.warning(", ".join(diff["modified_components"]))
        except Exception as e:
            st.error(f"Error comparing revisions: {e}")

def main(enable_debug: bool = False) -> None:
    """Main function for the Streamlit app."""
    # Set page config
    set_page_config()

    # Start debugger on localhost:5678, and ensure listen only once
    initialize_debugger(enable_debug)

    # Side bar configuration for llm and embeddings
    llm_model = st.sidebar.selectbox(
        "Select Model",
        options=[
            "qwen2.5-coder:1.5b",
            "llama3.2:1b",
            "llama3.1",
            "llama3.2",
            "llama3",
            "phi3",
            "openrouter/free",
            "openrouter/anthropic/claude-3.5-sonnet",
            "openrouter/meta-llama/llama-3-70b-instruct",
            "openrouter/mistralai/mixtral-8x7b-instruct",
            "openhermes",
            "llama2",
            "gpt-3.5-turbo"
        ]
    )

    openrouter_api_key = None
    if llm_model.startswith("openrouter/"):
        openrouter_api_key = st.sidebar.text_input(
            "OpenRouter API Key",
            type="password"
        )
        if not openrouter_api_key:
            st.sidebar.warning("Please provide an OpenRouter API Key to proceed.")

    embeddings_model = st.sidebar.selectbox(
        "Select Embeddings",
        options=[
            "all-minilm:l6-v2",
            "nomic-embed-text",
            "liquid/lfm-2.5-embedding-350m:free",
            "llama3",
            "openhermes",
            "phi3"
        ],
        help="When you change the embeddings model, documents will need to be reloaded into the matching collection.",
    )

    update_llm_config(llm_model, embeddings_model)

    # Add a text input in the sidebar for custom prompt
    custom_prompt = st.sidebar.text_area(
        "Enter system instruction:",
        placeholder=(
            "I want you to be a good assistant...\n"
            "You need to think like a human...\n"
        )
    )

    # create instance of assistant
    rag_assistant = get_or_create_assistant(llm_model, embeddings_model, custom_prompt, openrouter_api_key)

    load_assistant_chat_history(rag_assistant)

    # update session content with user input and assistant response
    session_messages = st.session_state["messages"]
    update_session_content(session_messages, rag_assistant)

    load_knowledge_base(rag_assistant)

    # Tabs for structured analysis features
    tab1, tab2, tab3 = st.tabs(["Chat Interface", "Structured Architecture", "Revision Comparison"])

    if st.sidebar.button("Run Structured Analysis", help="Extracts components and checks consistency"):
        run_structured_analysis(rag_assistant, llm_model=llm_model, openrouter_api_key=openrouter_api_key)

    with tab2:
        if st.button("Execute Extraction"):
            run_structured_analysis(rag_assistant, llm_model=llm_model, openrouter_api_key=openrouter_api_key)

    with tab3:
        run_revision_comparison()

    load_assistant_storage(rag_assistant, llm_model, embeddings_model)

    if st.sidebar.button(
        "New Run",
        help="Reset the session and reload the assistant."
    ):
        restart_assistant()

if __name__ == "__main__":
    main(enable_debug=False)
