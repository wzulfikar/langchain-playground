import os
from langchain.agents import Tool, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.utilities import GoogleSearchAPIWrapper


def create_chain():
    """Create agent chain using Google Search"""
    search = GoogleSearchAPIWrapper
    tools = [
        Tool(
            name="Current Search",
            # TODO: `search.run` doesn't seem to work (missing arg for `query`)
            func=search.run,
            description="useful for when you need to answer questions about current events or the current state of the world"
        ),
    ]
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = OpenAI(temperature=0)
    agent_chain = initialize_agent(
        tools,
        llm,
        agent="conversational-react-description",
        verbose=os.environ.get("VERBOSE", "0") == "1",
        memory=memory
    )
    return agent_chain


def main():
    chain = create_chain()
    # Start chat loop
    while True:
        print("Human:")
        human_input = input()

        print("AI:")
        output = chain.run(input=human_input)
        print(output)


if __name__ == "__main__":
    main()
