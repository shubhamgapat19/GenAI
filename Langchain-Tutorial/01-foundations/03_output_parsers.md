# Output Parsers — Deep Dive Notes

## The Problem Output Parsers Solve

LLMs return **text**. Your application needs **structured data** (JSON, objects, lists, numbers).

```
LLM says: "The sentiment is positive with a confidence of 0.92"
You need: {"sentiment": "positive", "confidence": 0.92}
```

Output Parsers bridge this gap by:
1. **Instructing** the LLM on the expected format (via `format_instructions`)
2. **Parsing** the LLM's text response into Python objects
3. **Validating** the output matches the expected schema

---

## How Output Parsers Work

```
                    ┌─────────────────────────┐
                    │    Output Parser         │
                    │                         │
format_instructions │   "Return JSON with     │
    (to prompt) ←───│    keys: name, age"     │
                    │                         │
        parse() ────│   '{"name":"X","age":5}'│───→ {"name": "X", "age": 5}
    (from LLM)      │   text → Python object  │        (Python dict)
                    └─────────────────────────┘
```

Every parser has two key methods:
- `.get_format_instructions()` → string to inject into your prompt
- `.parse(text)` → converts LLM output to structured data

In LCEL chains, parsers are used as the last step: `prompt | llm | parser`

---

## Parser Types Overview

| Parser | Output Type | Use Case | Reliability |
|--------|-------------|----------|-------------|
| `StrOutputParser` | `str` | Plain text extraction | ★★★★★ |
| `JsonOutputParser` | `dict` | Flexible JSON | ★★★★☆ |
| `PydanticOutputParser` | Pydantic model | Validated structured data | ★★★★☆ |
| `CommaSeparatedListOutputParser` | `list[str]` | Simple lists | ★★★★★ |
| `StructuredOutputParser` | `dict` | Schema-defined JSON | ★★★☆☆ |
| `EnumOutputParser` | `str` (enum value) | Fixed choices | ★★★★☆ |

---

## 1. StrOutputParser

The simplest parser — extracts `.content` from `AIMessage`.

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

# Without parser
response = llm.invoke("Hello")  # AIMessage(content="Hi!", ...)

# With parser in chain
chain = prompt | llm | StrOutputParser()
response = chain.invoke({"topic": "AI"})  # "Hi!" (just the string)
```

**When to use:** When you just want text output and will handle any further parsing yourself.

**Key insight:** Without `StrOutputParser`, your chain returns an `AIMessage` object. With it, you get a clean `str`. This matters because downstream components expect specific types.

---

## 2. JsonOutputParser

Parses LLM output as JSON into a Python dict.

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Always respond with valid JSON."),
    ("human", """Extract person info from: "{text}"
    
{format_instructions}"""),
])

chain = prompt | llm | parser

result = chain.invoke({
    "text": "John is 30 years old and works as an engineer",
    "format_instructions": parser.get_format_instructions()
})

# result = {"name": "John", "age": 30, "job": "engineer"}
print(type(result))  # <class 'dict'>
```

### With Pydantic Schema (recommended)

You can guide the JSON structure with a Pydantic model:

```python
from pydantic import BaseModel, Field

class PersonInfo(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age")
    occupation: str = Field(description="Job title")

parser = JsonOutputParser(pydantic_object=PersonInfo)
# format_instructions will include the schema
```

**When to use:** When you need flexible JSON output but don't need strict validation.

---

## 3. PydanticOutputParser (Most Important for Production)

Returns a **validated Pydantic object** — type-safe, with field validation.

```python
from pydantic import BaseModel, Field, field_validator
from langchain_core.output_parsers import PydanticOutputParser

class ProductReview(BaseModel):
    """Structured product review."""
    product_name: str = Field(description="Name of the product")
    rating: float = Field(description="Rating from 1.0 to 5.0")
    pros: list[str] = Field(description="List of positive points")
    cons: list[str] = Field(description="List of negative points")
    buy_recommendation: bool = Field(description="Would you recommend buying?")
    
    @field_validator("rating")
    @classmethod
    def rating_must_be_valid(cls, v):
        if not 1.0 <= v <= 5.0:
            raise ValueError("Rating must be between 1.0 and 5.0")
        return v

parser = PydanticOutputParser(pydantic_object=ProductReview)

prompt = ChatPromptTemplate.from_messages([
    ("human", """Analyze this product review and extract structured data:

Review: "{review_text}"

{format_instructions}"""),
])

chain = prompt | llm | parser

result = chain.invoke({
    "review_text": "Love this laptop! Great battery, fast processor. Screen could be brighter though. Definitely worth buying.",
    "format_instructions": parser.get_format_instructions()
})

# result is a ProductReview object
print(result.product_name)        # "laptop"
print(result.rating)              # 4.5
print(result.pros)                # ["Great battery", "Fast processor"]
print(result.cons)                # ["Screen could be brighter"]
print(result.buy_recommendation)  # True
```

### What `format_instructions` Looks Like

```
The output should be formatted as a JSON instance that conforms to the JSON schema below.

Here is the output schema:
{"properties": {"product_name": {"description": "Name of the product", "type": "string"}, "rating": {"description": "Rating from 1.0 to 5.0", "type": "number"}, ...}}
```

**When to use:** Any time you need structured data in production. The validation catches LLM mistakes.

---

## 4. CommaSeparatedListOutputParser

Returns a Python `list[str]` from comma-separated LLM output.

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("human", "List 5 {category}.\n\n{format_instructions}"),
])

chain = prompt | llm | parser

result = chain.invoke({
    "category": "Indian cities",
    "format_instructions": parser.get_format_instructions()
})

# result = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata"]
print(type(result))  # <class 'list'>
```

**When to use:** Simple lists where items are plain strings.

---

## 5. Structured Output (LLM's Native Feature)

Modern LLMs (GPT-4o, Claude) support **structured output natively** — no parser needed:

```python
from pydantic import BaseModel

class CalendarEvent(BaseModel):
    title: str
    date: str
    duration_minutes: int
    attendees: list[str]

# Use .with_structured_output() — built into the model
structured_llm = llm.with_structured_output(CalendarEvent)

result = structured_llm.invoke(
    "Set up a 30 min meeting with Alice and Bob next Tuesday about Q3 planning"
)

# result is a CalendarEvent object — guaranteed by the API
print(result.title)     # "Q3 Planning"
print(result.attendees) # ["Alice", "Bob"]
```

### `.with_structured_output()` vs `PydanticOutputParser`

| Feature | `.with_structured_output()` | `PydanticOutputParser` |
|---------|----------------------------|----------------------|
| How it works | API-level enforcement (function calling) | Prompt instruction + text parsing |
| Reliability | ★★★★★ (guaranteed format) | ★★★★☆ (LLM might deviate) |
| Model support | OpenAI, Anthropic, Google | Any model |
| Streaming | Partial objects supported | Needs full response |
| Flexibility | Fixed schema only | Can handle variations |

**Recommendation:** Use `.with_structured_output()` when available. Fall back to `PydanticOutputParser` for models that don't support it.

---

## Error Handling — When Parsing Fails

LLMs can produce malformed output. Handle it:

### OutputFixingParser — Auto-retry with error feedback

```python
from langchain.output_parsers import OutputFixingParser

# Wraps another parser — if it fails, sends error back to LLM to fix
fixing_parser = OutputFixingParser.from_llm(parser=pydantic_parser, llm=llm)

# If the first parse fails, it will:
# 1. Send the malformed output + error message to the LLM
# 2. Ask the LLM to fix it
# 3. Try parsing again
```

### RetryWithErrorOutputParser — Retry with original prompt

```python
from langchain.output_parsers import RetryWithErrorOutputParser

retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=pydantic_parser,
    llm=llm,
)

# If parsing fails, it retries with:
# - Original prompt
# - The bad output
# - The error message
# This gives the LLM full context to fix the issue
```

### Manual try/except

```python
from langchain_core.exceptions import OutputParserException

try:
    result = parser.parse(llm_output)
except OutputParserException as e:
    print(f"Parse failed: {e}")
    # Fallback logic: return default, retry, or ask user
```

---

## Streaming with Parsers

Some parsers support streaming (partial results):

```python
# JsonOutputParser supports streaming!
chain = prompt | llm | JsonOutputParser()

async for partial in chain.astream({"topic": "AI"}):
    print(partial)
    # First chunk: {}
    # Next chunk: {"name": "AI"}
    # Next chunk: {"name": "AI", "description": "Art..."}
    # Final: {"name": "AI", "description": "Artificial Intelligence"}
```

**Parsers that support streaming:**
- `StrOutputParser` ✅ (each token)
- `JsonOutputParser` ✅ (partial JSON)
- `PydanticOutputParser` ❌ (needs full response)
- `.with_structured_output()` ✅ (partial objects)

---

## Best Practices

1. **Always include `{format_instructions}` in your prompt** — without it, the LLM guesses
2. **Use Pydantic for production** — free validation, type safety, IDE autocomplete
3. **Prefer `.with_structured_output()`** — more reliable than text parsing
4. **Add `Field(description=...)` to every field** — helps the LLM understand what to fill
5. **Use `field_validator` for business rules** — catch impossible values (negative age, etc.)
6. **Wrap with `OutputFixingParser` for resilience** — auto-fixes malformed output
7. **Test with edge cases** — empty inputs, very long inputs, adversarial inputs

---

## Common Mistakes

| Mistake | What Happens | Fix |
|---------|-------------|-----|
| Forgetting `format_instructions` | LLM returns free-form text, parser fails | Always include in prompt |
| Using `JsonOutputParser` without schema | LLM invents random keys | Pass `pydantic_object` |
| Not handling parse errors | App crashes on bad LLM output | Wrap with `OutputFixingParser` or try/except |
| Over-complex schemas | LLM gets confused, returns garbage | Keep schemas flat and simple |
| Missing Field descriptions | LLM doesn't know what fields mean | Always add `description` |

---

## Decision Flowchart

```
Do you need structured output?
├── No → StrOutputParser
└── Yes
    ├── Just a list? → CommaSeparatedListOutputParser
    └── Complex structure?
        ├── Model supports function calling? → .with_structured_output()
        └── No → PydanticOutputParser + OutputFixingParser
```


