from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import os
from sys import stdin
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


template = """Question: {question}


Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm = OpenAI()
llm_chain = LLMChain(prompt=prompt, llm=llm)

print("Please type your question on Ej: 'What is at the core of Popper's theory of science?'")
print("=============================================================")
print("Question : ")
line = stdin.readline().strip()
while line:
    question = line
    response = llm_chain.run(question)
    print("Response:")
    print(response)
    print("=============================================================")
    print("Question : ")
    line = stdin.readline().strip()

