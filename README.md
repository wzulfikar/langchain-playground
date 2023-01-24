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
