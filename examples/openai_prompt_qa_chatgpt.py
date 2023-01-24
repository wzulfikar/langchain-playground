# From https://ahmadrosid.com/blog/langchain-openai-tutorial

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain

llm = OpenAI(temperature=.7)
template = """
  You are a teacher in physics for High School student. Given the text of question, your job is to write an answer with an example.
  Question: {text}
  Answer:
"""
prompt_template = PromptTemplate(input_variables=["text"], template=template)
answer_chain = LLMChain(llm=llm, prompt=prompt_template)
answer = answer_chain.run("What is the formula for Gravitational Potential Energy (GPE)?")
print(answer)
