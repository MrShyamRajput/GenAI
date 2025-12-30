from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

#PROMPT ENGINEERING TECHNIQUES

#1.Constraint Prompting

prompt1='''{topic}
Rules:
1. max 5  points
2. Each POint<=5 words
3. Use hinglish
4. No code
'''

#2. Output Format Control
prompt2='''
Explain {topic}.
Return output strictly in json fomrat:
{{
"definit":"",
"Use case":"",
"pros":[],
"cons":[]
}}
'''
#3. Few-Shot Prompting
prompt3='''
example:
Q: What is python?
A: Python is a high level programming language.

now answer
Q: What is {topic}?
'''
llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3
)
promptt=PromptTemplate(
    input_variables=["topic"],
    template=prompt2
)

prompt=PromptTemplate(
    input_variable=["topic"],
    template=prompt3
)
res=llm.invoke(prompt.format(topic="Django?"))
print(res.content)