from typing import List, Literal
from typing_extensions import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Define the structure
class Review(TypedDict):
    product: str
    sentiment: Literal["Positive", "Negative", "Neutral"]
    score: Annotated[int, "Score out of 10"]
    summary: str

class ExtractionSchema(TypedDict):
    reviews: List[Review]

# Initialize with structured output
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
structured_llm = llm.with_structured_output(ExtractionSchema)

# Execute
text = "The Pixel 10 is great (9/10), but the case I bought feels cheap (3/10)."
result = structured_llm.invoke(text)

print(result['reviews'])