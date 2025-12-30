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