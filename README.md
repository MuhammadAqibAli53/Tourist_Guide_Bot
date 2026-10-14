# Tourist Guide Bot

A multilingual tourism-assistance prototype built with Gradio, information retrieval, and optional LLM services.

## Features
- Multi-language support including English and Urdu
- Sentence Transformer embeddings and FAISS retrieval
- Wikipedia and external web-data integrations
- Translation, text-to-speech, and speech-recognition components
- Optional Groq and Gemini integrations

## Technology
Python, Gradio, Sentence Transformers, FAISS, Groq, Google Gemini, gTTS, SpeechRecognition, and Wikipedia.

## Setup
```bash
python -m venv .venv
pip install -r requirements.txt
python app.py
```
Activate the virtual environment before installing. Where applicable, set `GROQ_API_KEY` and `GEMINI_API_KEY` in the environment. Never commit credentials.

## Notes
This is a prototype; external APIs and internet data can change or become unavailable. Verify travel facts against current primary sources.

## Maintainer
[Muhammad Aqib Ali](https://github.com/MuhammadAqibAli53)
