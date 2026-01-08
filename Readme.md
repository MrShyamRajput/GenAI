🚀 My GenAI Learning Journey (Google Gemini)

This repository documents my day-by-day journey of learning Generative AI using Google Gemini (FREE).
I am learning GenAI from scratch, focusing on fundamentals → real applications, using Python.

This README will be updated daily as I progress.


Day 1 – GenAI Setup & Prompt Basics (Google Gemini)
What I Learned Today

Today I started my GenAI journey using Google Gemini (FREE) and learned the basics required to interact with a Generative AI model using Python.

🔹 Topics Covered

What is Generative AI
What is a Prompt
Google Gemini API setup
Text generation using Gemini
Temperature parameter
Max output tokens
Prompt engineering basics
PromptTemplate using LangChain

⚙️ Environment Setup
1️⃣ Create Virtual Environment
python -m venv genai_env
genai_env\Scripts\activate

2️⃣ Install Required Libraries
pip install google-generativeai langchain langchain-google-genai python-dotenv

3️⃣ API Key Setup

Created .env file:

GOOGLE_API_KEY=your_api_key_here

🔑 Gemini Configuration (Python)
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

✨ Text Generation Using Gemini
model = genai.GenerativeModel("gemini-pro")
response = model.generate_content("Explain Django in simple words")
print(response.text)

🧠 What is a Prompt?

A prompt is the instruction or input given to the AI to generate a response.

Example:
Explain Django ORM for beginners in simple Hinglish using 5 bullet points.

🌡️ Temperature Parameter

Temperature controls randomness of AI responses.

model = genai.GenerativeModel(
    "gemini-pro",
    generation_config={"temperature": 0.3}
)


Low temperature → factual answers

High temperature → creative answers

📏 Max Output Tokens

Controls the length of the response.

model = genai.GenerativeModel(
    "gemini-pro",
    generation_config={
        "temperature": 0.3,
        "max_output_tokens": 150
    }
)

🔗 Using Gemini with LangChain
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.3
)

print(llm.invoke("What is REST API?").content)

📦 PromptTemplate (Reusable Prompts)
from langchain.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["topic"],
    template="""
    You are a teacher.
    Explain {topic} in simple Hinglish.
    Give only 5 bullet points.
    """
)

prompt = template.format(topic="Django ORM")
print(llm.invoke(prompt).content)



Day 2 – Learned BAout Tokens

1️⃣ Tokens (LangChain Context)
What are Tokens?

Tokens are the smallest text units processed by the LLM connected through LangChain.
Input tokens → your prompt + chat history
Output tokens → AI response
LangChain sends both to the model

Example:

"I love Django"
≈ 3–4 tokens

Max Output Tokens in LangChain

In LangChain, max_output_tokens limits how many tokens the model can generate.

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.3,
    max_output_tokens=150
)

Important Behavior ⚠️

Gemini does not pre-plan the response
It generates text continuously
When token limit is reached → response is truncated (cut)

👉 Low token limit can cause:

Incomplete sentences
Broken explanations
Poor user experience




Day 3 – Chaining in LangChain (LCEL)
📌 Topic: Chaining & Its Types (Sequential & Parallel)

Today I learned about chaining in LangChain using LCEL (LangChain Expression Language).
Chaining is one of the core concepts used to build complex GenAI workflows.

🔹 What is Chaining?

Chaining means connecting multiple components like:
PromptTemplate
LLM (Google Gemini)
Output Parser
so that the output of one step becomes the input of the next step automatically.

In LangChain, this is done using the pipe (|) operator, which is part of LCEL.

Prompt → Model → Parser → Next Prompt → Model

🔹 LCEL (LangChain Expression Language)
LCEL is a declarative way to build chains using runnable components.

Example:
chain = prompt | llm | StrOutputParser()

Here:
prompt creates input
llm generates output
parser converts output to string

🔹 Types of Chaining Learned Today
1️⃣ Sequential Chaining

In Sequential Chaining,
the output of one step is passed step-by-step to the next component.

Example Use Case:

Generate text
Summarize that text
Refine the summary

Flow:
Step 1 → Step 2 → Step 3

Example:
chain = prompt1 | llm | parser | prompt2 | llm | parser


✔ Output flows linearly
✔ Best for multi-step reasoning tasks

2️⃣ Parallel Chaining

In Parallel Chaining,
the same input is processed by multiple chains at the same time,
and the outputs are collected in a dictionary.

This is done using RunnableParallel.

Example Use Case:

Generate notes
Generate quiz
Merge both later

Flow:
           → Chain A (Notes)
Input →
           → Chain B (Quiz)

Example:
parallel_chain = RunnableParallel({
    "notes": prompt1 | llm | parser,
    "quiz":  prompt2 | llm | parser
})


✔ Runs chains simultaneously
✔ Output format:

{
  "notes": "...",
  "quiz": "..."
}

🔹 Important Learnings

StrOutputParser converts LLM output into a single string
RunnableParallel collects multiple outputs into a dictionary
Parser does not store variables, it only transforms output
LCEL automatically passes data between components
Prompt variable names must match input keys exactly


Note: You can check Code in Chain.py file


Day 4 – RUNNABLES

Simple definition:
Runnable = ek unit jo input leta hai aur output deta hai

Bas itna.
LLM, Prompt, Parser, Chain
👉 sab internally Runnable hi hote hain

Real-life analogy 🧩
Soch tu ek factory chala raha hai:
Raw material → Machine → Polishing → Packing

Har machine = Runnable
LangChain me hum in machines ko pipe (|) se jod dete hain

CORE IDEA (MOST IMPORTANT)

LangChain me sab kuch Runnable hai:

Component	     Runnable?
PromptTemplate:     ✅
LLM (Gemini)  :     ✅
OutputParser  :     ✅
Chain	      :     ✅
Parallel execution:	✅


BASIC RUNNABLE FLOW
Flow diagram:
Input
 ↓
PromptTemplate
 ↓
LLM
 ↓
OutputParser
 ↓
Final Output

🧩 TYPES OF RUNNABLES (EXAM + INTERVIEW)
1️⃣ RunnableSequence (default)
Jab tu | use karta hai
prompt | llm | parser
->utput ek ke baad ek flow hota

2️⃣ RunnableParallel (parallel execution)
Same input se multiple outputs
Use case:
Notes + Quiz
Summary + Keywords
Explanation + Examples