from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 1. Define the template with a placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"), # The "Reserved Spot"
    ("human", "{input}")
])

# 2. Prepare the data to fill the placeholder
history = [
    HumanMessage(content="Hi, I'm Dave."),
    AIMessage(content="Hello Dave! How can I help you today?")
]

# 3. Format the prompt
final_prompt = prompt.format_messages(
    chat_history=history, 
    input="What is my name?"
)

print(final_prompt)