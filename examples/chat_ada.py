# From https://langchain.readthedocs.io/en/latest/modules/memory/examples/chatgpt_clone.html

import os
from langchain import ConversationChain, OpenAI
from langchain.chains.conversation.memory import ConversationalBufferWindowMemory
from langchain.callbacks import get_openai_callback


def create_chain(verbose: bool = False):
    """Create chat with prompt template and window memory"""
    chatgpt_chain = ConversationChain(
        llm=OpenAI(temperature=0,
                   # See all models here: https://beta.openai.com/docs/models/gpt-3
                   model_name="text-ada-001",
                   n=2,
                   best_of=2),
        verbose=verbose,
        memory=ConversationalBufferWindowMemory(k=2),
    )
    return chatgpt_chain


def main():
    print("Preparing AI.. Press ctrl+c to exit the program")

    chain = create_chain(os.environ.get("VERBOSE") == "1")

    # Start chat loop
    while True:
        print("Human:")
        human_input = input()

        print("AI:")
        with get_openai_callback() as cb:
            output = chain.run(human_input)
            print("  total_token:", cb.total_tokens)
            print(output.strip())


if __name__ == "__main__":
    main()
