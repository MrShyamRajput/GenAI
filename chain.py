#chain is used to create the link between the old and new meassge to remember the context

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chat_models.memory import ConversationBufferMemory
from langchian.chians import LLMChain

llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3,
    max_output_token=150
)


prompt=PromptTemplate(
    input_variables=['history','input'],
    template='''
previsious conversation
{history}

user:{input}
assitant:
''' 
)

memory=ConversationBufferMemory()

chain=LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)

print(chain.invoke({'input':"my name i shyam"}["text"]))