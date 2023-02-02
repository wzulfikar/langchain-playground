from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores.faiss import FAISS
from langchain.chains import VectorDBQAWithSourcesChain
from langchain import OpenAI


def create_chain():
    with open('fixtures/state_of_the_union.txt') as f:
        state_of_the_union = f.read()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(state_of_the_union)

    embeddings = OpenAIEmbeddings()

    docsearch = FAISS.from_texts(texts, embeddings)

    # Add in a fake source information
    for i, d in enumerate(docsearch.docstore._dict.values()):
        d.metadata = {'source': f"{i}-pl"}

    chain = VectorDBQAWithSourcesChain.from_chain_type(
        OpenAI(temperature=0), chain_type="stuff", vectorstore=docsearch)

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
