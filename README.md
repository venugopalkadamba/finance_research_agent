<h1>KVG Finance Research Assistant</h1>
[Python Version](https://img.shields.io/badge/python-3.10.12-blue)

## About
---
KVG Finance Research Assistant is an AI-powered agent that leverages the Llama 3 model via the Groq API and the Yahoo Finance Python module to assist users in conducting qualitative stock research. The application is built using Streamlit.

## Demo
---
<video controls src="https://github.com/venugopalkadamba/finance_research_agent/blob/main/assets/application_demo.mp4" title="Title"></video>

## Steps to run the project
- Install the required dependencies using the below command:
```
pip install -r requirements.txt
```

- Set the API key and the model name in ```.env``` and ```agent.py``` files
- Run the below command:
```
streamlit run app.py
```