# AUTOSAR HLD Document Analysis Assistant: User Guide & Documentation

## 1. System Overview
The AUTOSAR Assistant is a local RAG-powered web application built with Streamlit, Phidata, Qdrant, and Ollama. It helps automotive engineers review High-Level Design (HLD) specifications, extract component inventories, run consistency checks, and compare document revisions.

## 2. Installation & Environment Setup
1. **Prerequisites**: Python 3.9+, Docker, Ollama (`qwen2.5-coder:1.5b`, `all-minilm:l6-v2`).
2. **Infrastructure**:
   - Start Qdrant vector database: `./cookbook/run_qdrant.sh`
   - Start PostgreSQL vector store: `./cookbook/run_pgvector.sh`
3. **Dependencies**:
   ```bash
   pip install -r cookbook/requirements.txt
   ```

## 3. Running the Application
Start the Streamlit frontend:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

## 4. Key Features & Usage
- **Chat Tab**: Upload AUTOSAR PDFs via sidebar, then ask natural language questions. Append `/bd` to generate PlantUML block diagrams.
- **Architecture Tab**: Run LLM-based component and port extraction, review consistency findings, download JSON/CSV inventories.
- **Revisions Tab**: Compare two exported JSON architecture inventories to identify added, removed, or modified components.
- **Evaluation Tab**: Run dataset evaluation against test questions and generate keyword coverage reports.
