from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chains import VectorDBQAWithSourcesChain
from langchain import OpenAI


class Predictor:
    def __init__(self, chain):
        self.chain = chain

    def predict(self, human_input):
        output = self.chain({"question": human_input},
                            return_only_outputs=True)
        return output["answer"]


def create_chain(source_text, _is_verbose=False):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(source_text)

    embeddings = OpenAIEmbeddings()

    print("[INFO] creating FAISS wrapper..")
    docsearch = FAISS.from_texts(texts, embeddings)

    # Add in a fake source information
    for i, d in enumerate(docsearch.docstore._dict.values()):
        d.metadata = {'source': f"{i}-pl"}

    print("[INFO] creating vector db chain")
    chain = VectorDBQAWithSourcesChain.from_chain_type(
        OpenAI(temperature=0), chain_type="stuff", vectorstore=docsearch)

    print("[INFO] chain is ready!")
    return Predictor(chain)
