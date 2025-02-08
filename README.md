# Cover letter AI bot backend
Small side project to get into LLM toolchains.

This project aims to create a chatbot helper AI that writes cover letters based on your resume, a link to a job posting and any extra information you add.

## Features

- [ ] Get basic chat functionality using [Gemeni API]( https://ai.google.dev/gemini-api/docs/ )
- [ ] Add more possible LLM's (I might setup my own for example)
- [ ] Create a PDF reader/processing engine in [LlamaIndex]( https://github.com/run-llama/llama_index )
- [ ] Create a webscraper to read job postings using [crawl4ai]( https://github.com/unclecode/crawl4ai )
- [ ] Set up a database with the following features:
    - [ ] Keep track of chat logs
    - [ ] Keep track of uploaded PDF's and their parsed output, to reduce processing costs
- [ ] Expose endpoints for chatting using [FastAPI](https://github.com/fastapi/fastapi)
