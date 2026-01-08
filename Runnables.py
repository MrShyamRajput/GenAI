#related to the file: Runnables.py


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7
)

parser=StrOutputParser()

'''prompt=PromptTemplate(
    input_variables=['name','lang','domain'],
    template="Generative the Short poem of 2 to 3 paragrapj using the {name},{domain} and {lang} for the developer who should be impressed after reading this "
)
#This is a Runnable sequence
chian= prompt | llm | parser
response=chian.invoke({'name':"shyam", 'domain':"python",'lang':"hinglish"})

print(response)'''


#This is a Runnable Parallel

prompt1=PromptTemplate(
    input_variables=['topic','purpose'],
    template="Genearate the current information about the {topic} for the {purpose}"
)

prompt2=PromptTemplate(
    input_variables=['text'],
    template="Generate the 10 question and answer for the {topic} {purpose}"
)

Parallel_chian=RunnableParallel({
    "info": prompt1 | llm | parser,
    "quize": prompt2 | llm |parser
}
)

result=Parallel_chian.invoke({'topic':"job market of the python developer in pune",'purpose':'for the freshers who are looking to enter in it sector'})


print(result['info'])
