# Project Synopsis: AUTOSAR HLD Document Analysis Assistant

## 1. Executive Summary
The **AUTOSAR HLD Document Analysis Assistant** is an AI-powered engineering knowledge management and document analysis platform designed for automotive software architects, integration engineers, and testers. It ingests complex AUTOSAR High-Level Design (HLD) and Software Specification (SWS) documents, converts them into searchable vector embeddings, extracts structured architecture models (components, ports, interfaces), validates design consistency, and enables evidence-backed natural language Q&A.

## 2. Problem Statement & Context
Automotive software architectures are documented in extensive High-Level Design specifications that span hundreds of pages. Manual review is slow, prone to fragmentation across sections and tables, and lacks automated traceability and consistency validation. This system bridges the gap by transforming unstructured PDFs into a queryable, structured architecture knowledge base under strict human oversight.

## 3. Project Objectives
- **Accelerate Document Review**: Reduce manual lookup time across lengthy AUTOSAR SWS/SRS PDFs.
- **Structured Extraction**: Extract software components, ports, and interfaces into machine-readable JSON/CSV inventories.
- **Grounded Q&A**: Provide natural-language answers backed by source citations and context snippets.
- **Revision Comparison & Validation**: Automatically compare document revisions and flag consistency gaps (e.g., missing interfaces, unmatched ports).

## 4. Proposed Solution & Architecture
- **Frontend**: Streamlit-based interactive user interface with multi-tab navigation (Chat, Architecture, Revisions, Evaluation).
- **RAG & Orchestration**: Phidata framework integrated with Qdrant vector database and Ollama local LLMs (`qwen2.5-coder:1.5b`).
- **Parsing & Ingestion**: Custom `AutosarLoader` preserving page coordinates and metadata.
- **Validation & Compliance**: Algorithmic consistency checking and revision diffing.

## 5. Expected Outcomes & Deliverables
- Fully functional local RAG and structured architecture extraction pipeline.
- Automated evaluation harness with keyword coverage scoring.
- Comprehensive capstone documentation and folder structure ready for academic and industrial review.
