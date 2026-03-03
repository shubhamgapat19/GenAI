from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

chat_template=ChatPromptTemplate([
     ('system',"you are a helpful {domain} assistant."),
     ('human',"Explain in simple terms what is {topic}.")
    ])

prompt=chat_template.invoke(
    {
        'domain':'quantum physics',
        'topic':'wormhole'
    }
)

print(prompt)
