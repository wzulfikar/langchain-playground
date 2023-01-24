# Langchain Playground

Setup:

```sh
pip install virtualenv # optional, only run if you don't have virtualenv yet
asdf reshim python # optional, only run if using asdf and virtualenv not found
virtualenv .virtualenv # create virtual env
source .virtualenv/bin/activate # activate virtual env
pip install -r requirements.txt # install apps
```

- Copy .env.sample to .env and fill the blanks

Run examples:

- python examples/openai_prompt_qa_chatgpt.py

Notes:

- to update requirements.txt, run `pip freeze > requirements.txt`

## Docker

- Expose OpenAI api key to environment variable: `export OPENAI_API_KEY=sk-...`
- Run using docker: `docker run --rm -it -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY wzulfikar/langchain-playground`
- Send request to llm endpoint: `curl localhost:8000/api/llm?text=Explain+what+is+Github+in+less+than+80+words`

Preview:

![image](https://user-images.githubusercontent.com/7823011/214354481-9f72ded8-0763-4333-8ac4-cd60904b4ff4.png)
