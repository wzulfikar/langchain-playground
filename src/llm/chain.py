from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chains.conversation.memory import ConversationalBufferWindowMemory


def create_chain(template: str, verbose: False):
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
