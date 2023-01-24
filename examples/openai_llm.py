import sys
from langchain.llms import OpenAI

llm = OpenAI(temperature=.7)

# # Use first agument as text. Example:
# python examples/openai_llm.py "What is the formula for Gravitational Potential Energy (GPE)?"
text = sys.argv[1]

print("Asking AI..")
print(llm(text))
