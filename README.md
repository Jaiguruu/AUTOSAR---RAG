# TATA Techpulse AUTOSAR Assistant

This is an AUTOSAR document assistant with RAG, structured architecture analysis, and revision comparison.

## Environmental requirements

- **Operating System**: Linux/macOS (recommended) or Windows
- **Docker**: Required for vector stores
- **Python**: 3.9.6 (compatible; newer versions may work but have not been tested)

## Installation

### 1. Install Ollama and pull models

```shell
# Install Ollama (see https://github.com/ollama/ollama)
# Then pull the LLM and embedding models
ollama pull qwen-2.5-coder:1.5b
ollama pull all-minilm:l6-v2
```

### 2. Create a virtual environment

```shell
python3 -m venv ~/.venvs/autosar
source ~/.venvs/autosar/bin/activate
```

### 3. Install Python dependencies

```shell
pip install -r cookbook/requirements.txt
```

### 4. Install Docker

> Follow Docker's official installation guide: https://docs.docker.com/get-started/get-docker/

### 5. Run PgVector (PostgreSQL with vector extension)

You can start it with the helper script or directly via Docker:

```shell
# Using helper script
./cookbook/run_pgvector.sh
```

Or manually:

```shell
docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  phidata/pgvector:16
```

### 6. Run Qdrant (vector store)

```shell
# Using helper script
./cookbook/run_qdrant.sh
```

Or manually:

```shell
docker run -itd --name=qdrant \
    -p 6333:6333 \
    -p 6334:6334 \
    -e "QDRANT__SERVICE__API_KEY=123456" \
    -e "QDRANT__SERVICE__JWT_RBAC=true" \
    -v /home/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

> **Note**: If you encounter Docker image download errors, try switching to a different Docker registry mirror.

## Instant Start Guide

Assuming Docker and Ollama are installed, follow these steps to get the RAG app running quickly:

1. **Start Qdrant** (vector store):

   ```bash
   ./cookbook/run_qdrant.sh
   ```

2. **Start Ollama and pull models** (if not already done):

   ```bash
   ollama pull qwen-2.5-coder:1.5b
   ollama pull all-minilm:l6-v2
   ```

3. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

4. Open your browser at **http://localhost:8501**.

   - Upload your AUTOSAR PDFs via the **Upload** tab.
   - Ask questions directly.

### Example PDF

You can test with the sample PDF provided in the repository:

```
./cookbook/data/AUTOSAR/AUTOSAR_CP_SWS_DiagnosticCommunicationManager-54-78.pdf
```

---

## Running the RAG App

```shell
streamlit run app.py
```

- Open [http://localhost:8501](http://localhost:8501) to view the app.
- Add PDFs or ask questions directly.
- Example PDF path: `./cookbook/data/AUTOSAR/AUTOSAR_CP_SWS_DiagnosticCommunicationManager-54-78.pdf`


### 2. Create a virtual environment

```shell
python3 -m venv ~/.venvs/aienv
source ~/.venvs/aienv/bin/activate
```

### 3. Install libraries

```shell
pip install -r cookbook/requirements.txt
```

### 4. Install Docker

> Install docker according to this doc [Docker](https://docs.docker.com/get-started/get-docker/)

### 5. Run PgVector

- Run using a helper script

```shell
./cookbook/run_pgvector.sh
```

- OR run using the docker run command

```shell
docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  phidata/pgvector:16
```
### 6. Run Qdrant 

- Run using a helper script

```shell
./cookbook/run_qdrant.sh
```
- OR run using the docker run command

```shell
docker run -itd --name=qdrant \
    -p 6333:6333 \
    -p 6334:6334 \
    -e "QDRANT__SERVICE__API_KEY=123456" \
    -e "QDRANT__SERVICE__JWT_RBAC=true" \
    -v /home/qdrant_storage:/qdrant/storage:z \
        qdrant/qdrant
```

NOTE:

If following error occured in the step 5 or step 6, please try to change the mirror of docker

> docker: error pulling image configuration: download failed after attempts=6: dial tcp 199.16.156.11:443: connect: connection refused.


## Instant Start Guide

Follow these steps to get the RAG app up and running quickly (assumes Docker and Ollama are installed):

1. **Start Qdrant** (vector store):
   ```bash
   ./cookbook/run_qdrant.sh
   ```

2. **Start Ollama and pull models** (if not already):
   ```bash
   ollama pull llama3
   ollama pull nomic-embed-text
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

4. Open your browser at **http://localhost:8501**.  
   - Upload your PDFs via the **Upload** tab.  
   - Ask questions directly.

**Example PDF**: `./cookbook/data/AUTOSAR/AUTOSAR_CP_SWS_DiagnosticCommunicationManager-54-78.pdf`

---

### 7. Run RAG App

```shell
streamlit run app.py
```

- Open [localhost:8501](http://localhost:8501) to view your local RAG app.

- Add PDFs or ask question directly
- Example PDF: ./cookbook/data/AUTOSAR/AUTOSAR_CP_SWS_DiagnosticCommunicationManager-54-78.pdf 
"# AUTOSAR---RAG" 
