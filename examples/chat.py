# From https://langchain.readthedocs.io/en/latest/modules/memory/examples/chatgpt_clone.html

import os
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chains.conversation.memory import ConversationalBufferWindowMemory


def create_chain(template: str, verbose: bool = False):
    """Create chat with prompt template and window memory"""
    prompt = PromptTemplate(
        input_variables=["history", "human_input"],
        template=template
    )
    chatgpt_chain = LLMChain(
        llm=OpenAI(temperature=0),
        prompt=prompt,
        verbose=verbose,
        memory=ConversationalBufferWindowMemory(k=2),
    )
    return chatgpt_chain


def main():
    template = """Assistant is a large language model trained by OpenAI.
  Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
  Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
  Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
  {history}
  Human: {human_input}
  Assistant:"""

    print("Preparing AI.. Press ctrl+c to exit the program")

    chain = create_chain(template, os.environ.get("VERBOSE", False))

    # First input
    output = chain.predict(
        human_input="I want you to act as a Linux terminal. I will type commands and you will reply with what the terminal should show. I want you to only reply wiht the terminal output inside one unique code block, and nothing else. Do not write explanations. Do not type commands unless I instruct you to do so. When I need to tell you something in English I will do so by putting text inside curly brackets {like this}. My first command is echo Hello World!.")
    print("AI:")
    print(output)

    # Start chat loop
    while True:
        print("Human:")
        human_input = input()

        print("AI:")
        output = chain.predict(human_input=human_input)
        print(output)


if __name__ == "__main__":
    main()
