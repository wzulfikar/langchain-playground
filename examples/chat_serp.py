# From https://langchain.readthedocs.io/en/latest/modules/memory/examples/chatgpt_clone.html

import os
from langchain import OpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.callbacks import get_openai_callback


def create_chain():
    llm = OpenAI(temperature=0)

    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    agent = initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description",
        verbose=os.environ.get("VERBOSE") == "1")
    return agent


def main():
    print("Preparing AI.. Press ctrl+c to exit the program")

    chain = create_chain()

    # Start chat loop
    while True:
        print("Human:")
        human_input = input()

        print("AI:")
        with get_openai_callback() as cb:
            output = chain.run(human_input)
            print("  [INFO] total_token:", cb.total_tokens)
            print(output.strip())


if __name__ == "__main__":
    main()
