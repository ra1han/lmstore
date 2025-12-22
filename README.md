functions:
    - init: 
        parameter: openai client object
    - add: 
        Parameter: 
            - one (name and description) or multiple (json array with name and description) operator.
            - embedding model name
    - get:
        Parameter:
            - query: text
            - limit: int
            - llm_finalize: bool
            - model: text
    - export:
        parameter: file path or empty.
    - load:
        parameter: json with name, description, embedding 
