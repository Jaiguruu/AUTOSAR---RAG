"""Evaluation and answering script for AUTOSAR Large Data COM dataset."""

import os
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any

from phi.document import Document
from assistant.assistant import get_rag_assistant
from autosar_rag.autosar_loader import AutosarLoader
from autosar_rag.autosar_splitter import AutosarSplitter

WORKSPACE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = WORKSPACE_DIR / "autosar_large_data_com_test_dataset.json"
PDF_PATH = WORKSPACE_DIR / "assets" / "doc" / "AUTOSAR_SWS_LargeDataCOM.pdf"
OUTPUT_JSON = WORKSPACE_DIR / "autosar_large_data_com_answers.json"
OUTPUT_REPORT = WORKSPACE_DIR / "evaluation_report.md"


def index_document_if_needed(rag_assistant, pdf_path: Path):
    """Index PDF document into Qdrant vector database if collection is empty."""
    print(f"[*] Checking knowledge base for: {pdf_path.name}")
    try:
        vector_db = rag_assistant.knowledge_base.vector_db
        count = 0
        if hasattr(vector_db, "client") and hasattr(vector_db.client, "count"):
            count = vector_db.client.count(collection_name=vector_db.collection).count
        print(f"[*] Current vectors in collection '{vector_db.collection}': {count}")
    except Exception as e:
        print(f"[!] Note checking count: {e}")
        count = 0

    if count == 0:
        print(f"[*] Indexing {pdf_path.name} into knowledge base...")
        with open(pdf_path, "rb") as f:
            loader = AutosarLoader(f)
            splitter = AutosarSplitter(chunk_size=200, chunk_overlap=30)
            raw_docs = loader.load()
            split_docs = splitter.split_documents(raw_docs)

        phi_docs = [
            Document(
                content=doc.page_content,
                name=doc.metadata.get("source", pdf_path.name),
                meta_data=doc.metadata,
            )
            for doc in split_docs
        ]
        print(f"[*] Created {len(phi_docs)} document chunks. Storing into Qdrant...")
        rag_assistant.knowledge_base.load_documents(phi_docs, upsert=True)
        print("[+] Document successfully indexed into knowledge base.")
    else:
        print("[+] Knowledge base is already populated.")


def check_must_include(answer: str, must_include_str: str) -> Dict[str, Any]:
    """Check how many required terms/phrases are present in the answer."""
    terms = [t.strip() for t in must_include_str.split("|") if t.strip()]
    if not terms:
        return {"score": 1.0, "matched": [], "missing": []}

    answer_clean = re.sub(r"\s+", " ", answer.lower())
    matched = []
    missing = []
    for term in terms:
        term_clean = re.sub(r"\s+", " ", term.lower().strip())
        # Check direct substring
        if term_clean in answer_clean:
            matched.append(term)
            continue
        # Also check without underscores and spaces for identifier variations (e.g., PduR_LdComTransmit)
        condensed_term = re.sub(r"[\s_\*]+", "", term_clean)
        condensed_answer = re.sub(r"[\s_\*]+", "", answer_clean)
        if condensed_term and condensed_term in condensed_answer:
            matched.append(term)
            continue
        # Also check all individual key words if multi-word phrase
        words = [w for w in term_clean.split(" ") if len(w) > 2]
        if words and all(w in answer_clean for w in words):
            matched.append(term)
        else:
            missing.append(term)

    score = len(matched) / len(terms) if terms else 1.0
    return {
        "score": round(score, 2),
        "matched": matched,
        "missing": missing,
        "total_terms": len(terms),
    }


def answer_dataset(
    dataset_path: Path = DEFAULT_DATASET,
    llm_model: str = "qwen2.5-coder:1.5b",
    embeddings_model: str = "all-minilm:l6-v2",
    limit: int = None,
):
    """Run RAG assistant on dataset questions and evaluate results."""
    if not dataset_path.exists():
        print(f"[-] Dataset file not found at: {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    if limit:
        questions_data = questions_data[:limit]

    print(f"[+] Loaded {len(questions_data)} questions from {dataset_path.name}")
    print(f"[+] Initializing Assistant with LLM '{llm_model}' and embeddings '{embeddings_model}'...")

    rag_assistant = get_rag_assistant(
        llm_model=llm_model,
        embeddings_model=embeddings_model,
    )

    # Ensure document is indexed
    if PDF_PATH.exists():
        index_document_if_needed(rag_assistant, PDF_PATH)
    else:
        print(f"[!] Warning: PDF not found at {PDF_PATH}")

    results = []
    total_score = 0.0

    print("\n" + "=" * 80)
    print("STARTING DATASET ANSWERING & EVALUATION")
    print("=" * 80)

    for idx, item in enumerate(questions_data, start=1):
        q_id = item.get("id", f"Q-{idx}")
        category = item.get("category", "general")
        difficulty = item.get("difficulty", "medium")
        question = item.get("question", "")
        expected = item.get("expected_answer", "")
        must_include = item.get("must_include", "")
        source_pages = item.get("source_pages", "")

        print(f"\n[{idx}/{len(questions_data)}] [{q_id}] ({category} / {difficulty})")
        print(f"Question: {question}")

        # Query the assistant
        answer = ""
        try:
            for delta in rag_assistant.run(question):
                if isinstance(delta, str):
                    answer += delta
                elif hasattr(delta, "content"):
                    answer += str(delta.content)
            answer = answer.strip()
        except Exception as e:
            answer = f"Error during generation: {e}"

        # Evaluate must_include criteria
        eval_result = check_must_include(answer, must_include)
        score = eval_result["score"]
        total_score += score

        status = "PASS" if score >= 0.8 else ("PARTIAL" if score > 0 else "FAIL")

        print(f"Status: {status} (Coverage: {len(eval_result['matched'])}/{eval_result['total_terms']})")
        print(f"Answer snippet: {answer[:180]}...")
        if eval_result["missing"]:
            print(f"Missing terms: {eval_result['missing']}")

        results.append({
            "id": q_id,
            "category": category,
            "difficulty": difficulty,
            "question": question,
            "source_pages": source_pages,
            "expected_answer": expected,
            "must_include": must_include,
            "generated_answer": answer,
            "coverage_score": score,
            "status": status,
            "matched_terms": eval_result["matched"],
            "missing_terms": eval_result["missing"],
        })

    avg_score = (total_score / len(questions_data)) * 100 if questions_data else 0

    # Save JSON results
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "llm_model": llm_model,
                "embeddings_model": embeddings_model,
                "total_questions": len(questions_data),
                "average_coverage_pct": round(avg_score, 2),
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Save Markdown report
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("# AUTOSAR Large Data COM Evaluation Report\n\n")
        f.write(f"- **LLM Model**: `{llm_model}`\n")
        f.write(f"- **Embedding Model**: `{embeddings_model}`\n")
        f.write(f"- **Total Questions**: {len(questions_data)}\n")
        f.write(f"- **Average Keyword Coverage**: {avg_score:.1f}%\n\n")
        f.write("## Detailed Answers & Scores\n\n")
        for res in results:
            f.write(f"### {res['id']}: {res['question']}\n\n")
            f.write(f"- **Category**: {res['category']} | **Difficulty**: {res['difficulty']}\n")
            f.write(f"- **Source Pages**: {res['source_pages']}\n")
            f.write(f"- **Status**: `{res['status']}` ({int(res['coverage_score'] * 100)}% coverage)\n\n")
            f.write(f"**Expected Answer:**\n> {res['expected_answer']}\n\n")
            f.write(f"**Generated Answer:**\n{res['generated_answer']}\n\n")
            f.write(f"**Matched Criteria:** `{', '.join(res['matched_terms']) or 'None'}`\n\n")
            if res["missing_terms"]:
                f.write(f"**Missing Criteria:** `{', '.join(res['missing_terms'])}`\n\n")
            f.write("---\n\n")

    print("\n" + "=" * 80)
    print(f"EVALUATION COMPLETE: Average keyword coverage: {avg_score:.1f}%")
    print(f"Output JSON saved to: {OUTPUT_JSON}")
    print(f"Output Markdown report saved to: {OUTPUT_REPORT}")
    print("=" * 80)


def answer_single_question(question_id: str, llm_model: str = "qwen2.5-coder:1.5b", embeddings_model: str = "all-minilm:l6-v2"):
    """Run evaluation for a single question ID from the dataset."""
    # Load dataset
    if not DEFAULT_DATASET.exists():
        print(f"[-] Dataset file not found at: {DEFAULT_DATASET}")
        return
    with open(DEFAULT_DATASET, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    # Find the question by ID
    item = next((i for i in questions_data if str(i.get("id")) == question_id), None)
    if not item:
        print(f"[-] Question ID {question_id} not found in dataset.")
        return

    # Initialize assistant
    rag_assistant = get_rag_assistant(llm_model=llm_model, embeddings_model=embeddings_model)

    # Ensure document is indexed
    if PDF_PATH.exists():
        index_document_if_needed(rag_assistant, PDF_PATH)
    else:
        print(f"[!] Warning: PDF not found at {PDF_PATH}")

    # Extract fields
    q_id = item.get("id", question_id)
    category = item.get("category", "general")
    difficulty = item.get("difficulty", "medium")
    question = item.get("question", "")
    expected = item.get("expected_answer", "")
    must_include = item.get("must_include", "")
    source_pages = item.get("source_pages", "")

    print(f"\n[Single] [{q_id}] ({category} / {difficulty})")
    print(f"Question: {question}")

    # Query the assistant
    answer = ""
    try:
        for delta in rag_assistant.run(question):
            if isinstance(delta, str):
                answer += delta
            elif hasattr(delta, "content"):
                answer += str(delta.content)
        answer = answer.strip()
    except Exception as e:
        answer = f"Error during generation: {e}"

    # Evaluate must_include criteria
    eval_result = check_must_include(answer, must_include)
    score = eval_result["score"]
    status = "PASS" if score >= 0.8 else ("PARTIAL" if score > 0 else "FAIL")

    print(f"Status: {status} (Coverage: {len(eval_result['matched'])}/{eval_result['total_terms']})")
    print(f"Answer snippet: {answer[:180]}...")
    if eval_result["missing"]:
        print(f"Missing terms: {eval_result['missing']}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AUTOSAR dataset with optional single-question mode.")
    parser.add_argument("--limit", type=int, help="Limit number of questions to process (batch mode)")
    parser.add_argument("--question-id", type=str, help="Run evaluation for a single question ID")
    args = parser.parse_args()

    if args.question_id:
        answer_single_question(args.question_id)
    else:
        limit_arg = args.limit if args.limit is not None else None
        answer_dataset(limit=limit_arg)
