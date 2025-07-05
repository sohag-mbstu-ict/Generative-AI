from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser

# llm = ChatOllama(model="gemma:2b")
# response = llm.invoke("Explain gravity in simple terms.")
# print(response.content)

llm = ChatOllama(model='gemma:2b')
# Without bind.
chain = (
    llm
    | StrOutputParser()
)

output = chain.invoke("Repeat quoted words exactly: 'One two three four five.'")
print(output)
print("-------------------------------------------------------------------------")
# Output is 'One two three four five.'

# With bind.
chain = (
    llm.bind(stop=["three"])
    | StrOutputParser()
)

output = chain.invoke("Repeat quoted words exactly: 'One two three four five.'")
print(output)
print("-------------------------------------------------------------------------")
# Output is 'One two'


