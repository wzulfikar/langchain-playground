import os
from langchain import OpenAI, LLMMathChain


def create_chain():
    llm = OpenAI(temperature=0)
    llm_math = LLMMathChain(llm=llm,
                            verbose=os.environ.get('VERBOSE') == "1")

    return llm_math


def main():
    chain = create_chain()

    # Start chat loop
    while True:
        print("Human:")
        human_input = input()

        print("AI:")
        output = chain.run(human_input)
        print(output)


if __name__ == '__main__':
    main()
