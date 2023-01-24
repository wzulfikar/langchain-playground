from langchain.llms import OpenAI

llm = OpenAI()


def predict(text: str):
    print("Asking AI..")
    output = llm(text).strip()

    # TODO: error handling
    return {"ok": True, "output": output}
