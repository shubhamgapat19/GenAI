from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

# 1. Define your structure
class InventoryItem(TypedDict):
    """Information about a product in a warehouse."""
    name: str
    quantity: int
    category: Annotated[str, "The general type of item (e.g., Electronics, Food)"]

# 2. Initialize the model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 3. Create the structured LLM
structured_llm = llm.with_structured_output(InventoryItem)

# 4. Invoke
result = structured_llm.invoke("We just received 50 boxes of organic apples for the produce section.")

print(result)
# Output: {'name': 'organic apples', 'quantity': 50, 'category': 'Food'}