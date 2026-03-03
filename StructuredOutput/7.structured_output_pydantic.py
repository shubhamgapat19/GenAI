from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# 1. Define the Strict Schema
class ReviewItem(BaseModel):
    product: str = Field(description="Name of the product found in text.")
    sentiment: str = Field(description="Must be 'Positive', 'Negative', or 'Neutral'.")
    score: int = Field(ge=1, le=10, description="Score from 1 to 10.")

class ReviewAnalysis(BaseModel):
    """Container for multiple extracted reviews."""
    reviews: List[ReviewItem]

# 2. Initialize Model with Strict Structured Output
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
structured_llm = llm.with_structured_output(ReviewAnalysis, strict=True)

# 3. Invoke
text = "The headphones are a 9/10, but the charger is useless (1/10)."
result = structured_llm.invoke(text)

# The result is now a fully validated Pydantic object
for review in result.reviews:
    print(f"Product: {review.product}, Score: {review.score}")