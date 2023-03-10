import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chains import VectorDBQAWithSourcesChain
from langchain import OpenAI


def create_chain():
    # Provide `FILEPATH` env to supply a custom file
    filepath = os.environ.get('FILEPATH', 'fixtures/state_of_the_union.txt')

    print("[INFO] loading file:", filepath)
    with open(filepath) as f:
        state_of_the_union = f.read()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(state_of_the_union)

    embeddings = OpenAIEmbeddings()

    print("[INFO] creating FAISS wrapper..")
    docsearch = FAISS.from_texts(texts, embeddings)

    # Add in a fake source information
    for i, d in enumerate(docsearch.docstore._dict.values()):
        d.metadata = {'source': f"{i}-pl"}

    print("[INFO] creating vector db chain")
    chain = VectorDBQAWithSourcesChain.from_chain_type(
        OpenAI(temperature=0), chain_type="stuff", vectorstore=docsearch)

    print("[INFO] ready!")
    return chain


def main():
    chain = create_chain()
    # Start chat loop
    while True:
        print("Human:")
        human_input = input()

        print("AI:")
        output = chain({"question": human_input}, return_only_outputs=True)
        print(output)


if __name__ == "__main__":
    main()
