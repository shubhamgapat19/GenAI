"""
Phase 6: Production Patterns — Evaluation & Testing
=====================================================
Systematically test and evaluate LLM applications.

Topics covered:
- Unit testing chains
- LLM-as-judge evaluation
- Evaluation datasets
- Automated test suites
- Regression testing
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ============================================================
# 1. The Evaluation Problem
# ============================================================

# LLM outputs are non-deterministic. You can't use assertEqual().
# Solutions:
#   - Keyword/pattern matching (basic)
#   - LLM-as-judge (scalable)
#   - Human evaluation (ground truth)
#   - Statistical metrics (RAGAS, etc.)

print("=== The Evaluation Challenge ===")
print("Problem: LLM outputs vary each run")
print("Solution: Multiple evaluation strategies combined")
print()

# ============================================================
# 2. Building a Test Chain to Evaluate
# ============================================================

# A simple QA chain we want to test
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer the question based ONLY on the context.
If the answer isn't in the context, say "I don't know."

Context: {context}"""),
    ("human", "{question}"),
])

qa_chain = qa_prompt | llm | StrOutputParser()

# ============================================================
# 3. Basic Evaluation — Keyword Matching
# ============================================================

def keyword_eval(response: str, expected_keywords: list[str]) -> dict:
    """Check if response contains expected keywords."""
    response_lower = response.lower()
    found = [kw for kw in expected_keywords if kw.lower() in response_lower]
    missing = [kw for kw in expected_keywords if kw.lower() not in response_lower]
    score = len(found) / len(expected_keywords) if expected_keywords else 0
    return {"score": score, "found": found, "missing": missing}


test_cases = [
    {
        "context": "LangChain was created by Harrison Chase in October 2022.",
        "question": "Who created LangChain?",
        "expected_keywords": ["Harrison Chase"],
    },
    {
        "context": "Python supports multiple paradigms: OOP, functional, and procedural.",
        "question": "What paradigms does Python support?",
        "expected_keywords": ["OOP", "functional", "procedural"],
    },
    {
        "context": "The capital of France is Paris.",
        "question": "What is the capital of Germany?",
        "expected_keywords": ["don't know", "not in"],
    },
]

print("=== Keyword Evaluation ===")
for tc in test_cases:
    response = qa_chain.invoke({"context": tc["context"], "question": tc["question"]})
    result = keyword_eval(response, tc["expected_keywords"])
    status = "✅" if result["score"] >= 0.5 else "❌"
    print(f"  {status} Q: {tc['question'][:40]}... Score: {result['score']:.0%}")
    if result["missing"]:
        print(f"     Missing: {result['missing']}")
print()

# ============================================================
# 4. LLM-as-Judge Evaluation
# ============================================================

class EvalResult(BaseModel):
    """Evaluation result from LLM judge."""
    score: float = Field(ge=0, le=1, description="Score from 0.0 to 1.0")
    reasoning: str = Field(description="Brief explanation of the score")
    passed: bool = Field(description="Whether this passes quality threshold")


eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(EvalResult)

eval_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an evaluation judge. Score the AI's response on:
1. Correctness: Is the answer factually correct based on context?
2. Completeness: Does it fully answer the question?
3. Grounding: Does it ONLY use information from the context?

Score 0.0-1.0 where:
- 1.0 = Perfect (correct, complete, grounded)
- 0.7+ = Good (mostly correct, minor issues)
- 0.5 = Mediocre (partially correct)
- <0.5 = Bad (incorrect or hallucinated)"""),
    ("human", """Context: {context}
Question: {question}
AI Response: {response}

Evaluate the response."""),
])

eval_chain = eval_prompt | eval_llm


def llm_judge(context: str, question: str, response: str) -> EvalResult:
    """Use LLM to evaluate a response."""
    return eval_chain.invoke({
        "context": context,
        "question": question,
        "response": response,
    })


print("=== LLM-as-Judge Evaluation ===")
for tc in test_cases:
    response = qa_chain.invoke({"context": tc["context"], "question": tc["question"]})
    evaluation = llm_judge(tc["context"], tc["question"], response)
    status = "✅" if evaluation.passed else "❌"
    print(f"  {status} Q: {tc['question'][:40]}...")
    print(f"     Score: {evaluation.score:.2f} | {evaluation.reasoning[:60]}...")
print()

# ============================================================
# 5. Evaluation Dataset & Batch Testing
# ============================================================

# A proper evaluation dataset
eval_dataset = [
    {
        "context": "GPT-4 has a 128K token context window. It was released in March 2023.",
        "question": "What is GPT-4's context window?",
        "ground_truth": "128K tokens",
        "category": "factual",
    },
    {
        "context": "LangChain supports OpenAI, Anthropic, Google, and local models via Ollama.",
        "question": "Can I use LangChain with Anthropic?",
        "ground_truth": "Yes, LangChain supports Anthropic.",
        "category": "yes/no",
    },
    {
        "context": "Vector stores index documents by their embeddings for similarity search.",
        "question": "How do vector stores find relevant documents?",
        "ground_truth": "By comparing embedding similarity between the query and stored documents.",
        "category": "explanation",
    },
    {
        "context": "RAG combines retrieval with generation. It first finds relevant docs, then generates answers.",
        "question": "What is the weather today?",
        "ground_truth": "I don't know / not answerable from context.",
        "category": "out_of_scope",
    },
    {
        "context": "Agents use ReAct pattern: reason about the task, take action, observe results, repeat.",
        "question": "What pattern do LangChain agents use?",
        "ground_truth": "ReAct (Reason + Act)",
        "category": "factual",
    },
]


def run_evaluation_suite(chain, dataset: list) -> dict:
    """Run full evaluation suite and return metrics."""
    results = {
        "total": len(dataset),
        "passed": 0,
        "failed": 0,
        "avg_score": 0.0,
        "by_category": {},
        "failures": [],
    }
    
    scores = []
    
    for item in dataset:
        response = chain.invoke({
            "context": item["context"],
            "question": item["question"],
        })
        
        evaluation = llm_judge(item["context"], item["question"], response)
        scores.append(evaluation.score)
        
        category = item["category"]
        if category not in results["by_category"]:
            results["by_category"][category] = {"passed": 0, "total": 0}
        results["by_category"][category]["total"] += 1
        
        if evaluation.passed:
            results["passed"] += 1
            results["by_category"][category]["passed"] += 1
        else:
            results["failed"] += 1
            results["failures"].append({
                "question": item["question"],
                "expected": item["ground_truth"],
                "got": response[:100],
                "score": evaluation.score,
            })
    
    results["avg_score"] = sum(scores) / len(scores) if scores else 0
    return results


print("=== Full Evaluation Suite ===")
metrics = run_evaluation_suite(qa_chain, eval_dataset)
print(f"Results: {metrics['passed']}/{metrics['total']} passed ({metrics['avg_score']:.0%} avg)")
print(f"By category:")
for cat, data in metrics["by_category"].items():
    print(f"  {cat}: {data['passed']}/{data['total']}")
if metrics["failures"]:
    print(f"Failures:")
    for f in metrics["failures"]:
        print(f"  ❌ {f['question'][:40]}... (score: {f['score']:.2f})")
print()

# ============================================================
# 6. Regression Testing (compare versions)
# ============================================================

def compare_versions(chain_v1, chain_v2, dataset: list) -> dict:
    """Compare two chain versions on the same dataset."""
    v1_scores = []
    v2_scores = []
    regressions = []
    
    for item in dataset:
        r1 = chain_v1.invoke({"context": item["context"], "question": item["question"]})
        r2 = chain_v2.invoke({"context": item["context"], "question": item["question"]})
        
        e1 = llm_judge(item["context"], item["question"], r1)
        e2 = llm_judge(item["context"], item["question"], r2)
        
        v1_scores.append(e1.score)
        v2_scores.append(e2.score)
        
        if e2.score < e1.score - 0.2:  # Significant regression
            regressions.append({
                "question": item["question"],
                "v1_score": e1.score,
                "v2_score": e2.score,
            })
    
    return {
        "v1_avg": sum(v1_scores) / len(v1_scores),
        "v2_avg": sum(v2_scores) / len(v2_scores),
        "regressions": regressions,
        "improved": sum(v2_scores) / len(v2_scores) > sum(v1_scores) / len(v1_scores),
    }


# Example: compare strict vs lenient prompts
strict_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Answer ONLY from context. Say 'I don't know' if not in context.\n\nContext: {context}"),
        ("human", "{question}"),
    ]) | llm | StrOutputParser()
)

lenient_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Answer the question using the context as primary source. Context: {context}"),
        ("human", "{question}"),
    ]) | llm | StrOutputParser()
)

print("=== Regression Testing (v1 strict vs v2 lenient) ===")
comparison = compare_versions(strict_chain, lenient_chain, eval_dataset[:3])
print(f"  V1 (strict) avg: {comparison['v1_avg']:.2f}")
print(f"  V2 (lenient) avg: {comparison['v2_avg']:.2f}")
print(f"  V2 improved? {'Yes' if comparison['improved'] else 'No'}")
if comparison["regressions"]:
    for r in comparison["regressions"]:
        print(f"  ⚠️ Regression: {r['question'][:40]} ({r['v1_score']:.2f} → {r['v2_score']:.2f})")
