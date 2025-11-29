# https://python.langchain.com/api_reference/mistralai/chat_models/langchain_mistralai.chat_models.ChatMistralAI.html#langchain_mistralai.chat_models.ChatMistralAI

from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel
import os
os.environ["MISTRAL_API_KEY"] = "EN4au1y5Rsf03XAf03ObfvKVFTmz2f8e"


class AnswerWithJustification(BaseModel):
    '''An answer to the user question along with justification for the answer.'''

    answer: str
    justification: str

llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
structured_llm = llm.with_structured_output(
    AnswerWithJustification, include_raw=True
)

output = structured_llm.invoke(
    "What weighs more a pound of bricks or a pound of feathers"
)
print("output : ",output)
print("----------------------------------------------------------------------------")
print(output["parsed"])
print("----------------------------------------------------------------------------")
print("Answer:", output["parsed"].answer)
print("Justification:", output["parsed"].justification)

# -> {
#     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
#     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
#     'parsing_error': None
# }