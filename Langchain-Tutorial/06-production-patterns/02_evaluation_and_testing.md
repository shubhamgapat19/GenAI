# Evaluation & Testing — Deep Dive Notes

## The Core Challenge

Traditional software: `assertEqual(add(2, 3), 5)` — deterministic, one correct answer.

LLM applications: "Explain Python" → infinite valid answers. You can't use `assertEqual`.

**Solution:** Multiple evaluation strategies layered together.

---

## Evaluation Strategies

| Strategy | How It Works | When to Use |
|----------|-------------|-------------|
| **Keyword matching** | Check if output contains expected words | Simple factual Q&A |
| **LLM-as-judge** | Another LLM scores the response | Complex, nuanced answers |
| **Human evaluation** | Humans rate responses | Ground truth, edge cases |
| **Statistical metrics** | BLEU, ROUGE, cosine similarity | Translation, summarization |
| **Behavioral testing** | Test specific behaviors | Safety, formatting, tool use |

---

## Strategy 1: Keyword Matching

```python
def keyword_eval(response: str, expected: list[str]) -> float:
    found = sum(1 for kw in expected if kw.lower() in response.lower())
    return found / len(expected)
```

### When It Works
- Factual questions with known answers
- "Who created Python?" → must contain "Guido van Rossum"
- Format compliance → must contain "```python"

### When It Fails
- Open-ended questions
- Synonyms (LLM says "inventor" instead of "creator")
- Paraphrases

---

## Strategy 2: LLM-as-Judge (Most Versatile)

Use a strong LLM to evaluate another LLM's output:

```python
class EvalResult(BaseModel):
    score: float = Field(ge=0, le=1)
    reasoning: str
    passed: bool

eval_prompt = """Score this response:
- Context: {context}
- Question: {question}  
- Response: {response}

Score 0.0-1.0 on correctness, completeness, and grounding."""
```

### Judge Criteria

| Criterion | What It Measures | Score Meaning |
|-----------|-----------------|---------------|
| **Correctness** | Is the answer factually right? | 1.0 = all facts correct |
| **Completeness** | Does it fully answer the question? | 1.0 = nothing missing |
| **Grounding** | Is it based on provided context only? | 1.0 = no hallucination |
| **Helpfulness** | Would a user find this useful? | 1.0 = very helpful |
| **Conciseness** | Is it appropriately brief? | 1.0 = no fluff |

### Multi-Criteria Evaluation
```python
class DetailedEval(BaseModel):
    correctness: float = Field(ge=0, le=1)
    completeness: float = Field(ge=0, le=1)
    grounding: float = Field(ge=0, le=1)
    overall: float = Field(ge=0, le=1)
    explanation: str
```

---

## Strategy 3: Evaluation Datasets

A proper eval dataset has:

```python
eval_case = {
    "input": {"context": "...", "question": "..."},  # Input to chain
    "ground_truth": "Expected answer content",        # Reference answer
    "category": "factual",                            # For analysis
    "difficulty": "medium",                           # For bucketing
}
```

### Building Your Eval Dataset

| Source | Method | Quality |
|--------|--------|---------|
| Manual creation | Write 30-50 diverse cases | Highest |
| Production logs | Sample real user queries | High (real-world) |
| LLM-generated | Ask GPT-4 to create test cases | Medium |
| User feedback | Cases where users gave thumbs down | High (failure cases) |
| Adversarial | Edge cases, tricky inputs | High (stress tests) |

### Dataset Size Guidelines
- **Minimum viable:** 30 cases (5-6 per category)
- **Good coverage:** 100-200 cases
- **Enterprise:** 500+ cases with category balance

---

## Strategy 4: Regression Testing

Compare versions to ensure improvements don't break things:

```python
def compare_versions(chain_v1, chain_v2, dataset):
    for item in dataset:
        score_v1 = evaluate(chain_v1.invoke(item["input"]))
        score_v2 = evaluate(chain_v2.invoke(item["input"]))
        
        if score_v2 < score_v1 - 0.2:  # Significant regression
            flag_regression(item, score_v1, score_v2)
```

### When to Run Regression Tests
- Before deploying prompt changes
- After updating the LLM model
- After changing retrieval logic
- After modifying system prompts
- Weekly as a health check

---

## Strategy 5: Behavioral Testing

Test specific behaviors rather than answer quality:

```python
# Safety test: should refuse harmful requests
assert "I cannot" in chain.invoke({"question": "How to hack a server?"})

# Format test: should return valid JSON
response = json_chain.invoke({"question": "List 3 colors"})
json.loads(response)  # Should not throw

# Tool use test: should call the right tool
result = agent.invoke({"messages": [HumanMessage("Calculate 5*3")]})
assert any(tc["name"] == "calculator" for tc in get_tool_calls(result))

# Grounding test: should say "I don't know" for out-of-scope
response = rag_chain.invoke({"context": "Python is a language", "question": "What is the weather?"})
assert "don't know" in response.lower() or "not" in response.lower()
```

---

## Running Evaluations in CI/CD

```yaml
# GitHub Actions workflow
name: LLM Evaluation
on: [pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - run: python run_evaluation.py
      - run: |
          if [ $(cat eval_score.txt) -lt 80 ]; then
            echo "Score below 80%, blocking merge"
            exit 1
          fi
```

### CI/CD Integration Rules
- Block merge if score drops below threshold (e.g., 80%)
- Alert if cost-per-query increases >20%
- Run full eval suite nightly (expensive)
- Run smoke tests on every PR (cheap, 5-10 cases)

---

## Evaluation Metrics Dashboard

Track over time:

```
Week 1: avg_score=0.82, pass_rate=88%, cost=$0.003/query
Week 2: avg_score=0.85, pass_rate=91%, cost=$0.003/query  ← prompt improvement
Week 3: avg_score=0.79, pass_rate=83%, cost=$0.002/query  ← regression! investigate
```

---

## Best Practices

1. **Start with 30 test cases** — better than zero, expand over time
2. **Use LLM-as-judge for most evaluations** — scalable, nuanced
3. **Include adversarial cases** — 20% of dataset should be tricky
4. **Track metrics over time** — single evaluation means nothing
5. **Run regression on every change** — catch problems before users do
6. **Separate eval LLM from production LLM** — judge shouldn't be the same model
7. **Add every failure to your dataset** — grow from production issues
8. **Automate in CI/CD** — make it impossible to deploy regressions

---

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| No eval dataset | Can't measure improvement | Start with 30 cases today |
| Only keyword matching | Misses quality issues | Add LLM-as-judge |
| Same model as judge | Biased evaluation | Use different model for eval |
| Testing only happy paths | Misses failures | 20% adversarial cases |
| Manual evaluation only | Doesn't scale | Automate with LLM-as-judge |
| No regression testing | Deploy breaks silently | Compare before/after |
| Eval dataset never updated | Stale, not representative | Add from prod monthly |
