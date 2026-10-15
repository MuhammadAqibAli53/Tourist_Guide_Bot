# Tourist Guide Bot
### Multilingual AI · Information Retrieval · Speech Interfaces

A travel-assistance prototype exploring how multilingual NLP, semantic retrieval, and external information sources can work together in a single user-facing application.

**Skills showcased:** Python · NLP · Retrieval Systems · LLM API Integration · Speech Processing · Data Integration · Gradio

## Engineering highlights
| Challenge | Approach | Skill |
|---|---|---|
| Cross-language interaction | Language mappings and translation services | Internationalization, NLP |
| Retrieve relevant knowledge | Multilingual Sentence Transformers with FAISS | Vector search |
| Pull travel context | Wikipedia, requests, and Beautiful Soup integrations | Data retrieval |
| Conversational generation | Optional Groq and Gemini clients through environment keys | LLM integration |
| Voice accessibility | SpeechRecognition and gTTS components | Speech I/O |
| Application UX | Gradio interface | AI product prototyping |

## System overview
```text
User input (text / audio)
       |
Language processing / retrieval
       |
Knowledge sources + embedding search
       |
Optional generative model
       |
Text output / speech services
```

## Implementation details
The project initializes a multilingual sentence embedding model and optionally configures LLM clients when the corresponding API keys are available. It integrates multiple external packages, so availability and behavior depend on the runtime environment and providers.

## Tech stack
Python, Gradio, Sentence Transformers, FAISS, Wikipedia, Groq, Gemini, Google translation tools, SpeechRecognition, gTTS.

## Getting started
```bash
pip install -r requirements.txt
python app.py
```
Configure `GROQ_API_KEY` or `GEMINI_API_KEY` when using corresponding providers. Never commit credentials.

## Limitations
Travel information may be stale, external services may fail, and model outputs require verification. This is a prototype, not a certified travel-information service.

**Maintained by [Muhammad Aqib Ali](https://github.com/MuhammadAqibAli53).**
