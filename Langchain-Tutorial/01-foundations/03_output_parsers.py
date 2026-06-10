"""
Phase 1: Foundations — Output Parsers
======================================
Learn how to get structured data from LLM responses.

Topics covered:
- StrOutputParser (plain text)
- JsonOutputParser (JSON)
- PydanticOutputParser (validated Python objects)
- Comma-separated list parser
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ============================================================
# 1. StrOutputParser — Extract just the text
# ============================================================

prompt = ChatPromptTemplate.from_messages([
    ("human", "What is {concept} in one sentence?"),
])

# Without parser: returns AIMessage object
raw_response = llm.invoke(prompt.invoke({"concept": "embeddings"}))
print(f"Raw response type: {type(raw_response)}")
print(f"Raw response: {raw_response}")
print()

# With StrOutputParser: returns just the string
chain = prompt | llm | StrOutputParser()
clean_response = chain.invoke({"concept": "embeddings"})
print(f"Parsed response type: {type(clean_response)}")
print(f"Parsed response: {clean_response}")
print()

# ============================================================
# 2. JsonOutputParser — Get JSON output
# ============================================================

json_parser = JsonOutputParser()

json_prompt = ChatPromptTemplate.from_messages([
    ("system", "Always respond in valid JSON format."),
    ("human", """Extract information about this tech stack:
    
"We use Python with FastAPI for backend, React for frontend, and PostgreSQL for database"

Return JSON with keys: backend, frontend, database

{format_instructions}"""),
])

chain = json_prompt | llm | json_parser

result = chain.invoke({"format_instructions": json_parser.get_format_instructions()})
print("JSON parsed result:")
print(f"  Type: {type(result)}")  # dict!
print(f"  Backend: {result['backend']}")
print(f"  Frontend: {result['frontend']}")
print(f"  Database: {result['database']}")
print()

# ============================================================
# 3. PydanticOutputParser — Validated structured output
# ============================================================

# Define the expected output structure
class MovieReview(BaseModel):
    """Structured movie review."""
    title: str = Field(description="The movie title")
    rating: float = Field(description="Rating out of 10")
    genre: str = Field(description="Primary genre")
    summary: str = Field(description="One-line summary")
    recommend: bool = Field(description="Whether you'd recommend it")


from langchain_core.output_parsers import PydanticOutputParser

pydantic_parser = PydanticOutputParser(pydantic_object=MovieReview)

review_prompt = ChatPromptTemplate.from_messages([
    ("human", """Write a review for the movie "{movie}".

{format_instructions}"""),
])

chain = review_prompt | llm | pydantic_parser

review = chain.invoke({
    "movie": "Inception",
    "format_instructions": pydantic_parser.get_format_instructions()
})

print("Pydantic parsed result:")
print(f"  Type: {type(review)}")  # MovieReview object!
print(f"  Title: {review.title}")
print(f"  Rating: {review.rating}")
print(f"  Genre: {review.genre}")
print(f"  Summary: {review.summary}")
print(f"  Recommend: {review.recommend}")
print()

# ============================================================
# 4. CommaSeparatedListOutputParser
# ============================================================

list_parser = CommaSeparatedListOutputParser()

list_prompt = ChatPromptTemplate.from_messages([
    ("human", "List 5 popular {category}. {format_instructions}"),
])

chain = list_prompt | llm | list_parser

result = chain.invoke({
    "category": "Python web frameworks",
    "format_instructions": list_parser.get_format_instructions()
})

print("List parsed result:")
print(f"  Type: {type(result)}")  # list!
print(f"  Items: {result}")
