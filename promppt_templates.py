from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

llm=ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0.3
)

template='''
You are my froiend, explain all the topics in simple and one one paragraph
1. what is {skills}
2. Applications 0f {skills}
3.what we can do with {skills}
'''
prompt=PromptTemplate(
    input_variable=["skills"],
    template=template
)

res=llm.invoke(prompt.format(skills="jwt auth"))
print(prompt.format(skills="jwt auth"))
print(res.content)