# Conversation Memory Strategies — Deep Dive Notes

## The Core Problem

LLMs have fixed context windows. A long conversation eventually exceeds them:

```
GPT-4o-mini: 128K tokens context
Average message: ~100 tokens
Max conversation: ~1200 messages before overflow

BUT: Long context = slow + expensive + attention dilution
Practical limit: 20-50 messages before quality degrades
```

**Solution:** Memory strategies that keep conversations manageable without losing important information.

---

## Strategy Comparison

| Strategy | Keeps | Loses | Token Usage | Best For |
|----------|-------|-------|-------------|----------|
| **Buffer** (full) | Everything | Nothing | Grows unbounded | Short conversations |
| **Window** (last N) | Recent context | Old messages | Fixed | Chat interfaces |
| **Token Buffer** | Last N tokens | Oldest messages | Fixed cap | Cost control |
| **Summary** | Key facts | Details, nuance | Small, fixed | Long-running sessions |
| **Combined** | Summary + recent | Old details | Moderate | Production apps |

---

## Strategy 1: Buffer Memory (Full History)

Keep every message. Simplest but most expensive.

```python
# Just pass all messages to the LLM
messages = [all_previous_messages] + [new_message]
response = llm.invoke(messages)
```

**When to use:** Conversations under 20 messages, when every detail matters.
**When NOT to use:** Long conversations, high-traffic apps, cost-sensitive.

---

## Strategy 2: Window Memory (Last N)

Keep only the most recent N messages:

```python
def window_memory(messages, window_size=10):
    return messages[-window_size:]
```

**Pros:** Simple, predictable cost, O(1) memory
**Cons:** Forgets old information completely

**Typical window sizes:**
- 6 messages (3 turns) — chatbot with short answers
- 10-20 messages — technical assistant
- 20-40 messages — complex multi-step tasks

---

## Strategy 3: Token Buffer

Smarter than window — trims by token count instead of message count:

```python
from langchain_core.messages import trim_messages

trimmed = trim_messages(
    messages,
    max_tokens=1000,
    strategy="last",        # Keep latest messages
    token_counter=llm,      # Use LLM's tokenizer
    include_system=True,    # Always keep system prompt
    allow_partial=False,    # Don't cut messages in half
)
```

### trim_messages Parameters

| Parameter | Purpose |
|-----------|---------|
| `max_tokens` | Maximum total tokens to keep |
| `strategy` | `"last"` (keep newest) or `"first"` (keep oldest) |
| `token_counter` | LLM model or callable for counting |
| `include_system` | Always preserve system message |
| `allow_partial` | Whether to truncate a message mid-text |
| `start_on` | `"human"` ensures history starts with a user message |

---

## Strategy 4: Summary Memory

Use an LLM to summarize old messages into a compact form:

```python
summary_prompt = "Summarize this conversation in 2-3 sentences. Capture: user identity, topic, key decisions."
summary = llm.invoke(old_messages + [summary_prompt])

# Use summary as context
effective_messages = [
    SystemMessage(content=f"Conversation summary: {summary}"),
    *recent_messages,  # Last few messages in full
]
```

### What Makes a Good Summary
- User identity (name, role)
- Topic/project being discussed
- Key decisions or preferences stated
- Current problem/question thread

### Summary Trigger
- After every N messages
- When total tokens exceed threshold
- On explicit "new topic" detection

---

## Strategy 5: Combined (Summary + Window)

The production-grade approach. Best of both worlds:

```
[Summary of messages 1-50] + [Full messages 51-60]

Old context (compressed):  "User is Shubham, building GPS logistics platform with NestJS..."
Recent context (full):     Last 10 messages with all detail
```

```python
def combined_memory(messages, recent_count=10):
    if len(messages) <= recent_count:
        return messages  # No need to summarize yet
    
    old = messages[:-recent_count]
    recent = messages[-recent_count:]
    
    summary = summarize(old)
    
    return [
        SystemMessage(content=f"Previous context: {summary}"),
        *recent,
    ]
```

---

## Choosing the Right Strategy

```
Conversation < 10 turns? → Buffer (full history)
                          ↓ No
Cost is critical? → Window (last 6-10 messages)
                  ↓ No
Need old context? → Combined (summary + recent)
                  ↓ No
Simple chat app? → Token Buffer (trim_messages)
```

---

## Implementation Pattern

```python
class SmartMemory:
    def __init__(self, max_messages=20, summary_threshold=15):
        self.messages = []
        self.summary = ""
        self.max_messages = max_messages
        self.summary_threshold = summary_threshold
    
    def add(self, message):
        self.messages.append(message)
        if len(self.messages) > self.summary_threshold:
            self._compress()
    
    def _compress(self):
        # Summarize older half
        split = len(self.messages) // 2
        old = self.messages[:split]
        self.summary = summarize(old, existing_summary=self.summary)
        self.messages = self.messages[split:]
    
    def get_context(self):
        context = []
        if self.summary:
            context.append(SystemMessage(content=f"Context: {self.summary}"))
        context.extend(self.messages[-self.max_messages:])
        return context
```

---

## Best Practices

1. **Start with window memory** — simplest, works for most chatbots
2. **Add summary when users report "it forgot"** — that's your signal
3. **Always keep system prompt** — `include_system=True` in trim_messages
4. **Ensure history starts with HumanMessage** — some models require alternating roles
5. **Monitor token usage** — track how many tokens history consumes per request
6. **Test with long conversations** — simulate 50+ turns to verify strategy works
7. **Consider the UX** — users expect the AI to remember their name for the whole session

---

## Cost Analysis

```
Scenario: 1000 users, avg 30 messages/session, GPT-4o-mini ($0.15/1M input tokens)

Buffer (full):     30 msgs × 100 tokens × 30 calls = 90K tokens/session
                   × 1000 users = 90M tokens = $13.50/day

Window (last 10):  10 msgs × 100 tokens × 30 calls = 30K tokens/session
                   × 1000 users = 30M tokens = $4.50/day

Combined:          (200 summary + 10×100) × 30 calls = 36K tokens/session
                   × 1000 users = 36M tokens = $5.40/day

Savings: Buffer → Window = 67% cost reduction
```
