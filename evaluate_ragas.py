"""
Ragas Evaluation and Benchmarking Suite for Ashcombe & Rowe Estates FAQ RAG.
Evaluates:
1. Faithfulness (Zero hallucination against knowledge base)
2. Answer Relevance (Direct, concise addressing of user inquiry)
3. Context Precision (Top-1 ranking of relevant FAQ chunk)
4. Context Recall (Completeness of retrieved facts)
5. Out-of-Domain Zero-Hallucination Rate (Strict fallback to 020 7946 0100)
"""

import os
import sys
import json
import types
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Compatibility shim for Ragas on Python 3.13 / langchain-community
if "langchain_community.chat_models.vertexai" not in sys.modules:
    fake_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
    fake_vertex.ChatVertexAI = None
    sys.modules["langchain_community.chat_models.vertexai"] = fake_vertex

from src.rag_engine import faq_engine
from src.llm_factory import llm_client
from src.config import settings

# -------------------------------------------------------------------------
# Golden Evaluation Dataset (12 Representative Test Cases)
# -------------------------------------------------------------------------
GOLDEN_FAQ_EVAL_DATASET = [
    {
        "id": "TC-01",
        "question": "Can I pop into the Chelsea branch on a Sunday afternoon?",
        "expected_faq_ids": ["FAQ-023", "FAQ-029"],
        "ground_truth": "Chelsea branch is open on Sunday from 11:00 am to 3:00 pm. It is the only branch open on Sundays.",
        "category": "hours",
        "key_facts": ["11:00 am", "3:00 pm", "Sunday"],
    },
    {
        "id": "TC-02",
        "question": "Are any of your offices open on Sunday other than Chelsea?",
        "expected_faq_ids": ["FAQ-029"],
        "ground_truth": "Only our Chelsea branch is open on Sundays, from 11:00 am to 3:00 pm. All other branches and the head office are closed on Sundays.",
        "category": "hours",
        "key_facts": ["Chelsea", "closed on Sundays"],
    },
    {
        "id": "TC-03",
        "question": "What commission fee do you charge to sell a residential property?",
        "expected_faq_ids": ["FAQ-040"],
        "ground_truth": "Our standard sole-agency fee is 1.5% of the agreed sale price plus VAT. Multi-agency is 2.25% plus VAT. There are no upfront costs.",
        "category": "selling",
        "key_facts": ["1.5%", "VAT"],
    },
    {
        "id": "TC-04",
        "question": "Where is your head office located in London?",
        "expected_faq_ids": ["FAQ-009", "FAQ-004"],
        "ground_truth": "Our head office is at 22 Halden Mews, Mayfair, London W1K 2PX.",
        "category": "contact",
        "key_facts": ["Mayfair"],
    },
    {
        "id": "TC-05",
        "question": "Which local branch handles properties in the Kew area?",
        "expected_faq_ids": ["FAQ-020"],
        "ground_truth": "Kew is covered by our Richmond branch at 9 Hill Street, Richmond TW9 1SX (020 7946 0430).",
        "category": "branches",
        "key_facts": ["Richmond", "9 Hill Street"],
    },
    {
        "id": "TC-06",
        "question": "Is a property valuation free of charge?",
        "expected_faq_ids": ["FAQ-039"],
        "ground_truth": "Yes. Our market valuations are free and come with no obligation to instruct us.",
        "category": "selling",
        "key_facts": ["free", "no obligation"],
    },
    {
        "id": "TC-07",
        "question": "What paperwork and documents do I need to submit an offer?",
        "expected_faq_ids": ["FAQ-036"],
        "ground_truth": "To make an offer you will need photo ID, proof of address and proof of funds (or a mortgage agreement in principle) under UK anti-money laundering regulations.",
        "category": "buying",
        "key_facts": ["photo ID", "proof of address"],
    },
    {
        "id": "TC-08",
        "question": "Do you handle new-build homes priced under £1 million?",
        "expected_faq_ids": ["FAQ-006"],
        "ground_truth": "Our new-build service is for homes priced at £1 million and above. Our resale listings cover a wider range of prices, including some homes under £1 million.",
        "category": "company",
        "key_facts": ["£1 million and above"],
    },
    {
        "id": "TC-09",
        "question": "Do you offer lettings or property management services?",
        "expected_faq_ids": ["FAQ-044"],
        "ground_truth": "No. We focus on sales only and do not currently offer lettings or property management. For rental enquiries please call our head office on 020 7946 0100.",
        "category": "general",
        "key_facts": ["sales only", "020 7946 0100"],
    },
    {
        "id": "TC-10",
        "question": "Are you registered with any professional real estate bodies?",
        "expected_faq_ids": ["FAQ-007"],
        "ground_truth": "Yes. We are members of Propertymark and The Property Ombudsman, and we follow their codes of practice.",
        "category": "company",
        "key_facts": ["Propertymark", "Property Ombudsman"],
    },
    # Negative / Out-of-Domain Guardrail Tests
    {
        "id": "TC-11",
        "question": "Do you have commercial airplane hangars for sale in Manchester?",
        "expected_faq_ids": [],
        "ground_truth": f"I'm sorry, I don't have that information in my records. Please contact our head office on {settings.HEAD_OFFICE_PHONE}.",
        "category": "out_of_domain",
        "key_facts": [settings.HEAD_OFFICE_PHONE],
    },
    {
        "id": "TC-12",
        "question": "Can I get a discount policy on solar panels from David Rowe?",
        "expected_faq_ids": [],
        "ground_truth": f"I'm sorry, I don't have that information in my records. Please contact our head office on {settings.HEAD_OFFICE_PHONE}.",
        "category": "out_of_domain",
        "key_facts": [settings.HEAD_OFFICE_PHONE],
    },
]


def run_ragas_evaluation() -> Dict[str, Any]:
    """
    Executes Ragas evaluation pipeline across the golden test dataset.
    Computes both Ragas metrics and deterministic factual evaluation scores.
    """
    print("\n" + "=" * 80)
    print(" ASHCOMBE & ROWE ESTATES — RAGAS EVALUATION & BENCHMARK SUITE")
    print(" Knowledge Base: data/faq.csv (47 luxury real estate policies)")
    print(f" Embedding Model: {settings.EMBEDDING_MODEL}")
    print("=" * 80 + "\n")

    transcripts_dir = PROJECT_ROOT / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    questions: List[str] = []
    answers: List[str] = []
    contexts: List[List[str]] = []
    ground_truths: List[str] = []

    results_table: List[Dict[str, Any]] = []

    correct_retrievals = 0
    faithful_answers = 0
    zero_hallucination_out_of_domain = 0
    total_in_domain = 0
    total_out_of_domain = 0

    for tc in GOLDEN_FAQ_EVAL_DATASET:
        q = tc["question"]
        expected_ids = tc["expected_faq_ids"]
        gt = tc["ground_truth"]
        key_facts = tc["key_facts"]
        category = tc["category"]

        questions.append(q)
        ground_truths.append(gt)

        # 1. Retrieve from FAQ Engine
        faq_item = faq_engine.search_faq(q)

        if expected_ids:
            # In-Domain Query
            total_in_domain += 1
            if faq_item:
                retrieved_answer = faq_item.answer
                retrieved_context = [f"[{faq_item.faq_id}] {faq_item.question}: {faq_item.answer}"]
                retrieved_id = faq_item.faq_id
                
                # Check top-1 retrieval accuracy against acceptable valid FAQ items
                hit = (retrieved_id in expected_ids)
                if hit:
                    correct_retrievals += 1

                # Check factual accuracy
                factual_score = sum(1 for k in key_facts if k.lower() in retrieved_answer.lower()) / len(key_facts)
                if factual_score >= 0.75:
                    faithful_answers += 1
            else:
                retrieved_answer = f"Please contact our head office on {settings.HEAD_OFFICE_PHONE}."
                retrieved_context = ["No matching context found."]
                retrieved_id = None
                hit = False
                factual_score = 0.0

        else:
            # Out-of-Domain Query: Expected to NOT match FAQ and safely return head office phone
            total_out_of_domain += 1
            if faq_item is None:
                # Correctly triggered zero-hallucination threshold!
                retrieved_answer = f"I'm sorry, I don't have that information in my records. Please contact our head office on {settings.HEAD_OFFICE_PHONE}."
                retrieved_context = ["Out-of-domain query safely filtered by confidence threshold."]
                retrieved_id = "HEAD_OFFICE_FALLBACK"
                hit = True
                factual_score = 1.0
                zero_hallucination_out_of_domain += 1
            else:
                retrieved_answer = faq_item.answer
                retrieved_context = [faq_item.answer]
                retrieved_id = faq_item.faq_id
                hit = False
                factual_score = 0.0

        answers.append(retrieved_answer)
        contexts.append(retrieved_context)

        expected_str = "/".join(expected_ids) if expected_ids else "DEFLECTION"
        results_table.append({
            "test_id": tc["id"],
            "question": q,
            "category": category,
            "expected_id": expected_str,
            "retrieved_id": retrieved_id or "NONE",
            "hit": hit,
            "factual_faithfulness": round(factual_score, 2),
            "retrieved_answer": retrieved_answer,
        })

    # Summary Calculations
    hit_rate = round(correct_retrievals / total_in_domain, 4) if total_in_domain else 1.0
    faithfulness_rate = round(faithful_answers / total_in_domain, 4) if total_in_domain else 1.0
    zero_hallucination_rate = round(zero_hallucination_out_of_domain / total_out_of_domain, 4) if total_out_of_domain else 1.0
    overall_accuracy = round((correct_retrievals + zero_hallucination_out_of_domain) / len(GOLDEN_FAQ_EVAL_DATASET), 4)

    # -------------------------------------------------------------------------
    # Attempt Ragas Library Evaluation if LLM Available
    # -------------------------------------------------------------------------
    ragas_scores: Dict[str, float] = {}
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

        eval_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        # If LLM client is initialized, use it as the evaluator
        if llm_client._llm is not None:
            from ragas.llms import LangchainLLMWrapper
            ragas_llm = LangchainLLMWrapper(llm_client._llm)
            ragas_result = evaluate(
                eval_dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=ragas_llm,
                raise_exceptions=False,
            )
            ragas_scores = {k: round(float(v), 4) for k, v in ragas_result.items()}
    except Exception as e:
        # Gracefully handle when offline / local without OpenAI keys
        pass

    # Print Formatted Report Table
    print(f"{'ID':<6} | {'Category':<14} | {'Expected':<12} | {'Retrieved':<12} | {'Hit':<5} | {'Faithfulness':<12}")
    print("-" * 80)
    for r in results_table:
        status_sym = "✅" if r["hit"] else "❌"
        print(f"{r['test_id']:<6} | {r['category']:<14} | {str(r['expected_id']):<12} | {str(r['retrieved_id']):<12} | {status_sym:<5} | {r['factual_faithfulness']:<12}")

    print("\n" + "=" * 80)
    print(" 📊 RAGAS EVALUATION METRIC SUMMARY")
    print("=" * 80)
    print(f" • Context Precision (Hit@1 Retrieval Rate) : {hit_rate * 100:.1f}%")
    print(f" • Faithfulness (Factual Consistency Rate)   : {faithfulness_rate * 100:.1f}%")
    print(f" • Zero-Hallucination Rate (Out-of-Domain)  : {zero_hallucination_rate * 100:.1f}%")
    print(f" • Overall Pipeline Accuracy                : {overall_accuracy * 100:.1f}%")
    if ragas_scores:
        print("\n • Ragas Automated Metric Scores:")
        for metric_name, val in ragas_scores.items():
            print(f"    - {metric_name}: {val:.4f}")
    print("=" * 80 + "\n")

    # Export Report to JSON
    report_data = {
        "benchmark_name": "Ashcombe & Rowe Estates FAQ RAG Evaluation",
        "total_test_cases": len(GOLDEN_FAQ_EVAL_DATASET),
        "metrics": {
            "context_precision_hit_rate": hit_rate,
            "faithfulness_rate": faithfulness_rate,
            "zero_hallucination_out_of_domain_rate": zero_hallucination_rate,
            "overall_accuracy": overall_accuracy,
            "ragas_library_scores": ragas_scores,
        },
        "results": results_table,
    }

    json_path = transcripts_dir / "ragas_evaluation_report.json"
    md_path = transcripts_dir / "ragas_evaluation_report.md"
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # Export Report to Markdown
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Ashcombe & Rowe Estates — Ragas FAQ Evaluation Report\n\n")
            f.write(f"- **Total Test Cases**: {len(GOLDEN_FAQ_EVAL_DATASET)}\n")
            f.write(f"- **Context Precision (Hit@1)**: {hit_rate * 100:.1f}%\n")
            f.write(f"- **Faithfulness (Factual Consistency)**: {faithfulness_rate * 100:.1f}%\n")
            f.write(f"- **Zero-Hallucination Guardrail Rate**: {zero_hallucination_rate * 100:.1f}%\n")
            f.write(f"- **Overall Pipeline Accuracy**: {overall_accuracy * 100:.1f}%\n\n")
            f.write("## Test Case Breakdown\n\n")
            f.write("| ID | Category | Question | Expected ID | Retrieved ID | Hit | Faithfulness |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :---: | :---: |\n")
            for r in results_table:
                status = "✅" if r["hit"] else "❌"
                f.write(f"| {r['test_id']} | {r['category']} | {r['question']} | {r['expected_id']} | {r['retrieved_id']} | {status} | {r['factual_faithfulness']} |\n")

        print(f"✅ Ragas evaluation report exported to:")
        print(f"   • {json_path}")
        print(f"   • {md_path}\n")
    except OSError as e:
        print(f"[Notice: Could not write RAGAS report files to disk ({e}) - display summary above]\n")

    return report_data


if __name__ == "__main__":
    run_ragas_evaluation()
