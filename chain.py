'''#chain is used to create the link between the old and new meassge to remember the context

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3,
   
)


prompt1=PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="Generate a 5 pont summary on from the folloing text \n {text}",
    input_variables=["text"]
)

parser=StrOutputParser()


#This is a example of sequestial chain

chain= prompt1 | llm | parser | prompt2 | llm | parser

result=chain.invoke({"topic":"future of python developer"})
print(result)
chain.get_graph().print_ascii()
'''

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

model1= ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

model2=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite"
)

prompt1=PromptTemplate(
    template="Generate the short and simple note formt he followig topics \n {text}",
    input_variables=["text"]
)

prompt2=PromptTemplate(
    template="Generate the 5 short queston and answer from the folowinng text \n {text}",
    input_variables=["text"]
)

prompt3=PromptTemplate(
    template="Merge th provided note and quize into followinf into single document \n {notes} and {quize}",
    input_variables=["notes","quize"]
)

parser=StrOutputParser()

parallel_chian=RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quize": prompt2 | model1 | parser
}
)

prompt0=PromptTemplate(
    template="Give me the 5 paragraph each of 5 lines  information of: Future of job market of gen ai developers in pune"
)
start= prompt0 | model1 | parser
merge_chain = prompt3 | model1 | parser

chain =start | parallel_chian | merge_chain


result=chain.invoke({"text":start})
print(result)