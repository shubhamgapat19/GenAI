# Prompt Templates — Deep Dive Notes

## What is a Prompt Template?

A **Prompt Template** is a reusable blueprint for creating prompts with dynamic variables. Instead of hardcoding prompts as strings, you define a template once and fill in variables at runtime.

**Without templates (bad):**
```python
prompt = f"You are a {role}. Answer this: {question}"  # fragile, no structure
```

**With templates (good):**
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    ("human", "{question}"),
])
```

---

## Why Templates Matter

| Problem | Template Solution |
|---------|-------------------|
| Prompts scattered in code | Centralized, reusable templates |
| No type safety | Variables are validated |
| Hard to version control | Templates are data, not code |
| Copy-paste errors | Define once, use everywhere |
| Hard to test | Test template separately from LLM |
| No separation of concerns | Prompt engineering ≠ application logic |

---

## Types of Prompt Templates

### 1. ChatPromptTemplate (Most Common)

Creates structured chat messages with roles:

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Be {style}."),
    ("human", "{question}"),
])

# Invoke with variables
result = prompt.invoke({
    "role": "data scientist",
    "style": "concise",
    "question": "What is overfitting?"
})
```

**Output:** A `ChatPromptValue` containing formatted `messages` list.

### 2. PromptTemplate (Simple string template)

For cases where you just need a formatted string (not messages):

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "Summarize this text in {num_words} words:\n\n{text}"
)

result = prompt.invoke({"num_words": 50, "text": "Long article..."})
# Returns: StringPromptValue
```

### 3. MessagesPlaceholder

Injects a **dynamic list of messages** into the template. Critical for conversation history:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),  # ← dynamic messages
    ("human", "{input}"),
])
```

This is how memory works — you inject previous messages into the template.

---

## Template Syntax

### Variable Substitution
```python
# Single curly braces for variables
("human", "Tell me about {topic}")

# To use literal curly braces, double them
("human", "Format: {{key: value}}")  # outputs: {key: value}
```

### Multi-line Templates
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior developer.
Your job is to review code for:
- Bugs
- Performance issues
- Security vulnerabilities

Be specific and actionable."""),
    ("human", "Review this {language} code:\n\n```{language}\n{code}\n```"),
])
```

---

## Creating Templates — All Methods

### Method 1: from_messages() — Most flexible
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "System message here"),
    ("human", "User message with {variable}"),
    ("ai", "Example AI response"),      # For few-shot
    ("human", "Another {question}"),
])
```

### Method 2: Tuple shorthand
```python
# These are equivalent:
("system", "Hello")           # Tuple shorthand
SystemMessage(content="Hello")  # Explicit message object
```

### Method 3: from_template() — Simple single-message
```python
prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
# Creates a single HumanMessage template
```

---

## Advanced Features

### Partial Templates — Pre-fill some variables

When you know some variables early but others come later:

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Respond in {language}."),
    ("human", "{question}"),
])

# Fix the language now, fill the rest later
english_prompt = prompt.partial(language="English")
hindi_prompt = prompt.partial(language="Hindi")

# Later, only need role and question
english_prompt.invoke({"role": "tutor", "question": "What is AI?"})
```

**Use case:** Multi-tenant apps where some context is set at startup.

### Partial with Functions

```python
from datetime import datetime

def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Today's date is {date}. You are a helpful assistant."),
    ("human", "{question}"),
])

# date will be computed at invoke time
prompt = prompt.partial(date=get_current_date)
```

---

## Few-Shot Prompting with Templates

Teach the LLM by example:

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You classify customer feedback as positive, negative, or neutral."),
    
    # Few-shot examples
    ("human", "This product is amazing!"),
    ("ai", "positive"),
    ("human", "Worst purchase ever."),
    ("ai", "negative"),
    ("human", "It's okay I guess."),
    ("ai", "neutral"),
    
    # Actual input
    ("human", "{feedback}"),
])

# The LLM now understands the expected format from examples
```

### Dynamic Few-Shot (from a list)

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input": "I love it!", "output": "positive"},
    {"input": "Terrible.", "output": "negative"},
    {"input": "It's fine.", "output": "neutral"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# Use in a full prompt
full_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify sentiment."),
    few_shot_prompt,
    ("human", "{feedback}"),
])
```

---

## MessagesPlaceholder — Deep Dive

This is one of the most important concepts for building real apps:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("chat_history", optional=True),  # optional=True means it can be empty
    ("human", "{input}"),
])

# First message (no history)
result = prompt.invoke({
    "chat_history": [],
    "input": "My name is Shubham"
})

# Second message (with history)
result = prompt.invoke({
    "chat_history": [
        HumanMessage(content="My name is Shubham"),
        AIMessage(content="Hello Shubham! How can I help?"),
    ],
    "input": "What's my name?"
})
```

### Why not just concatenate strings?
- Message roles are preserved (system vs human vs AI)
- Token counting is accurate per role
- API expects structured messages, not raw text
- Some models treat roles differently (e.g., system messages have higher weight)

---

## Template Composition

Build complex prompts from smaller reusable parts:

```python
# Base system prompt
system_base = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Current date: {date}."),
])

# Add history
with_history = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Current date: {date}."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

# Different versions for different use cases
analyst_prompt = with_history.partial(role="data analyst", date="2026-05-24")
tutor_prompt = with_history.partial(role="patient tutor", date="2026-05-24")
```

---

## Pipeline Integration (with LCEL)

Templates are the **first step** in any chain:

```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
#       ↑        ↑         ↑
#   Template   Model    Parser
#   (format)   (think)  (extract)
```

The template's `.invoke()` output type matches what the LLM expects as input.

---

## Input Validation

Templates validate that all required variables are provided:

```python
prompt = ChatPromptTemplate.from_messages([
    ("human", "Tell me about {topic} in {language}"),
])

# This will raise an error — missing 'language'
prompt.invoke({"topic": "AI"})
# KeyError: 'language'

# Check required variables
print(prompt.input_variables)  # ['topic', 'language']
```

---

## Best Practices

1. **One template per task** — don't make mega-templates that do everything
2. **Use system messages** — they strongly influence LLM behavior
3. **Keep variables semantic** — `{user_question}` not `{x}`
4. **Use partial for shared context** — date, user info, app config
5. **Few-shot > instructions** — showing examples often works better than explaining
6. **Test templates independently** — `prompt.invoke(...)` without the LLM to verify formatting
7. **Version your prompts** — prompts are the most impactful part of your app
8. **Use MessagesPlaceholder for history** — never build conversation as a string

---

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| `f"You are {role}"` | Not reusable, no validation | Use `ChatPromptTemplate` |
| Putting history in system msg | Wastes system message influence | Use `MessagesPlaceholder` |
| Too many variables | Template becomes confusing | Split into composable templates |
| No few-shot examples | LLM guesses output format | Add 2-3 examples |
| Forgetting `{format_instructions}` | Output parsers don't work | Always include when using parsers |


