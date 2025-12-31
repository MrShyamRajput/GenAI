from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3,
    max_output_tokens=100
)

prompt=PromptTemplate(
    input_variables=['topic','tokens'],
    template="explaiin this topic {topic} only in {tokens}"
)

response=llm.invoke(prompt.format(topic="django",tokens=100)).content
print(response)


'''max_output_token does not force the answer , it only suggest the mdoel to generate the answer near the token limit , if we want to strictly generate the answer then we can mention in the template like in above example
'''