#Prompts= Role + Task +Content + Format

from langchain_google_genai import ChatGoogleGenerativeAI


llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.5
)

#prompt="Explain django"   Bad Prompt

prompt='''
You are a python instructor .
Explain Django orm for beginners.
Use simple Hinglish.
'''
print(llm.invoke(prompt).content)