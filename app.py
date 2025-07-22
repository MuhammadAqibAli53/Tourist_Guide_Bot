import gradio as gr
import os
import time
import logging
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")
from deep_translator import GoogleTranslator
# Core libraries
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import json

# Translation and TTS
from googletrans import Translator
import gtts
import io
import tempfile

# Web scraping and API calls
import requests
from bs4 import BeautifulSoup
import wikipedia
import urllib.parse

# Speech recognition
import speech_recognition as sr
from pydub import AudioSegment

# LLM API clients
import groq
from google.generativeai import GenerativeModel
import google.generativeai as genai

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TouristGuideBot:
    def __init__(self):
        """Initialize the Tourist Guide Bot with all necessary components."""
        
        # Initialize components
        self.translator = Translator()
        self.embedding_model = None
        self.faiss_index = None
        self.knowledge_base = []
        self.groq_client = None
        self.gemini_model = None
        
        # Supported languages
        self.languages = {
            "English": "en",
            "Urdu": "ur", 
            "Arabic": "ar",
            "French": "fr",
            "Spanish": "es",
            "German": "de",
            "Italian": "it",
            "Chinese": "zh",
            "Japanese": "ja",
            "Hindi": "hi"
        }
        
        # Initialize models and data
        self.setup_models()
        self.load_or_create_knowledge_base()
        
    def setup_models(self):
        """Setup embedding model and LLM APIs."""
        try:
            # Load multilingual sentence transformer
            self.embedding_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
            logger.info("✅ Embedding model loaded successfully")
            
            # Setup API clients (add your API keys)
            groq_api_key = os.getenv('GROQ_API_KEY')
            gemini_api_key = os.getenv('GEMINI_API_KEY')
            
            if groq_api_key:
                self.groq_client = groq.Groq(api_key=groq_api_key)
                logger.info("✅ Groq API client initialized")
            
            if gemini_api_key:
                genai.configure(api_key=gemini_api_key)
                self.gemini_model = GenerativeModel('gemini-pro')
                logger.info("✅ Gemini API client initialized")
                
        except Exception as e:
            logger.error(f"❌ Error setting up models: {e}")
    
    def load_or_create_knowledge_base(self):
        """Load existing knowledge base or create a new one."""
        kb_path = "travel_knowledge_base.pkl"
        index_path = "faiss_index.idx"
        
        try:
            # Try to load existing knowledge base
            if os.path.exists(kb_path) and os.path.exists(index_path):
                with open(kb_path, 'rb') as f:
                    self.knowledge_base = pickle.load(f)
                self.faiss_index = faiss.read_index(index_path)
                logger.info(f"✅ Loaded existing knowledge base with {len(self.knowledge_base)} chunks")
            else:
                # Create new knowledge base
                self.create_knowledge_base()
                
        except Exception as e:
            logger.error(f"❌ Error loading knowledge base: {e}")
            self.create_knowledge_base()
    
    def create_knowledge_base(self):
        """Create knowledge base from sample travel data."""
        logger.info("🔄 Creating new knowledge base...")
        
        # Sample travel knowledge (in practice, load from files/Wikipedia)
        sample_data = [
            {
                "content": "Paris is the capital city of France, known for its iconic Eiffel Tower, Louvre Museum, and romantic atmosphere. Best visited in spring or fall.",
                "location": "Paris, France",
                "category": "destination"
            },
            {
                "content": "Dubai offers luxury shopping, ultramodern architecture, and desert safaris. The Burj Khalifa is the world's tallest building.",
                "location": "Dubai, UAE", 
                "category": "destination"
            },
            {
                "content": "Tokyo combines traditional Japanese culture with cutting-edge technology. Visit temples, enjoy sushi, and experience the bustling city life.",
                "location": "Tokyo, Japan",
                "category": "destination"
            },
            {
                "content": "New York City offers Broadway shows, world-class museums, Central Park, and diverse neighborhoods like Times Square and Brooklyn.",
                "location": "New York, USA",
                "category": "destination"
            },
            {
                "content": "Always carry a universal adapter, pack light, research local customs, and keep copies of important documents when traveling internationally.",
                "location": "General",
                "category": "travel_tips"
            },
            {
                "content": "Book flights 2-3 months in advance for best prices. Use flight comparison websites and be flexible with dates.",
                "location": "General", 
                "category": "travel_tips"
            },
            {
                "content": "Istanbul bridges Europe and Asia, featuring the Blue Mosque, Hagia Sophia, and Grand Bazaar. Turkish cuisine is exceptional.",
                "location": "Istanbul, Turkey",
                "category": "destination"
            },
            {
                "content": "Rome offers ancient history with the Colosseum, Vatican City, Trevi Fountain, and delicious Italian food.",
                "location": "Rome, Italy",
                "category": "destination"
            }
        ]
        
        # Create embeddings for all content
        texts = [item["content"] for item in sample_data]
        embeddings = self.embedding_model.encode(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings.astype('float32'))
        
        # Store knowledge base
        self.knowledge_base = sample_data
        
        # Save to disk
        with open("travel_knowledge_base.pkl", 'wb') as f:
            pickle.dump(self.knowledge_base, f)
        faiss.write_index(self.faiss_index, "faiss_index.idx")
        
        logger.info(f"✅ Created knowledge base with {len(self.knowledge_base)} chunks")
    


    def translate_text(self, text: str, target_lang: str = "en", source_lang: str = "auto") -> str:
        try:
            translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
            return translated
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text

    
    def search_wikipedia(self, query: str, max_results: int = 3) -> List[str]:
        """Search Wikipedia for relevant travel information."""
        try:
            # Search Wikipedia
            search_results = wikipedia.search(query, results=max_results)
            summaries = []
            
            for title in search_results[:max_results]:
                try:
                    summary = wikipedia.summary(title, sentences=3)
                    summaries.append(f"Wikipedia - {title}: {summary}")
                except wikipedia.exceptions.DisambiguationError as e:
                    # Try the first option
                    try:
                        summary = wikipedia.summary(e.options[0], sentences=3)
                        summaries.append(f"Wikipedia - {e.options[0]}: {summary}")
                    except:
                        continue
                except:
                    continue
            
            return summaries
            
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return []
    
    def search_duckduckgo(self, query: str, max_results: int = 3) -> List[str]:
        """Search DuckDuckGo for travel information."""
        try:
            # DuckDuckGo instant answer API
            url = f"https://api.duckduckgo.com/"
            params = {
                'q': query + " travel guide",
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            results = []
            
            # Extract abstract
            if data.get('Abstract'):
                results.append(f"DuckDuckGo: {data['Abstract']}")
            
            # Extract related topics
            for topic in data.get('RelatedTopics', [])[:2]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append(f"DuckDuckGo: {topic['Text']}")
            
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []
    
    def retrieve_similar_chunks(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve similar chunks from knowledge base using FAISS."""
        try:
            # Create embedding for query
            query_embedding = self.embedding_model.encode([query])
            
            # Search in FAISS index
            distances, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
            
            # Get relevant chunks
            similar_chunks = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.knowledge_base):
                    chunk = self.knowledge_base[idx].copy()
                    chunk['similarity_score'] = float(distances[0][i])
                    similar_chunks.append(chunk)
            
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []
    
    def query_llm(self, prompt: str) -> str:
        """Query LLM using available APIs."""
        try:
            # Try Groq first
            if self.groq_client:
                try:
                    completion = self.groq_client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful and knowledgeable tourist guide assistant. Provide accurate, helpful, and engaging travel advice."
                            },
                            {
                                "role": "user", 
                                "content": prompt
                            }
                        ],
                        model="llama3-8b-8192",
                        temperature=0.7,
                        max_tokens=1024
                    )
                    return completion.choices[0].message.content
                except Exception as e:
                    logger.error(f"Groq API error: {e}")
            
            # Try Gemini as fallback
            if self.gemini_model:
                try:
                    response = self.gemini_model.generate_content(prompt)
                    return response.text
                except Exception as e:
                    logger.error(f"Gemini API error: {e}")
            
            # Fallback response if no API available
            return "I'm sorry, but I'm currently unable to access the AI models. Please check your API keys and try again."
            
        except Exception as e:
            logger.error(f"LLM query error: {e}")
            return "An error occurred while processing your request. Please try again."
    
    def process_audio_input(self, audio_file) -> str:
        """Convert speech to text."""
        if audio_file is None:
            return ""
            
        try:
            # Initialize recognizer
            r = sr.Recognizer()
            
            # Load audio file
            with sr.AudioFile(audio_file) as source:
                audio = r.record(source)
            
            # Convert speech to text
            text = r.recognize_google(audio)
            return text
            
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return "Could not understand audio. Please try again."
    
    def generate_speech(self, text: str, lang_code: str = "en") -> str:
        """Generate speech from text using gTTS."""
        try:
            tts = gtts.gTTS(text=text, lang=lang_code, slow=False)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                return tmp_file.name
                
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None
    
    def answer_question(self, question: str, language: str, enable_tts: bool = True) -> Tuple[str, Optional[str]]:
        """Main function to answer user questions."""
        try:
            lang_code = self.languages.get(language, "en")
            
            # Step 1: Translate question to English if needed
            if lang_code != "en":
                english_question = self.translate_text(question, target_lang="en")
            else:
                english_question = question
            
            # Step 2: Retrieve similar chunks from knowledge base
            similar_chunks = self.retrieve_similar_chunks(english_question, k=3)
            
            # Step 3: Get live web data
            wikipedia_results = self.search_wikipedia(english_question)
            duckduckgo_results = self.search_duckduckgo(english_question)
            
            # Step 4: Build context from all sources
            context_parts = []
            
            # Add knowledge base chunks
            if similar_chunks:
                context_parts.append("From Knowledge Base:")
                for chunk in similar_chunks:
                    context_parts.append(f"- {chunk['content']}")
            
            # Add Wikipedia results
            if wikipedia_results:
                context_parts.append("\nFrom Wikipedia:")
                for result in wikipedia_results:
                    context_parts.append(f"- {result}")
            
            # Add DuckDuckGo results
            if duckduckgo_results:
                context_parts.append("\nFrom DuckDuckGo:")
                for result in duckduckgo_results:
                    context_parts.append(f"- {result}")
            
            context = "\n".join(context_parts)
            
            # Step 5: Create prompt for LLM
            prompt = f"""Based on the following context, please answer the user's travel question comprehensively and helpfully.

Context:
{context}

User Question: {english_question}

Please provide a detailed, helpful answer that combines information from the context. Focus on practical travel advice, recommendations, and useful tips. Keep the response informative but conversational."""
            
            # Step 6: Query LLM
            english_answer = self.query_llm(prompt)
            
            # Step 7: Translate answer back to user's language
            if lang_code != "en":
                final_answer = self.translate_text(english_answer, target_lang=lang_code)
            else:
                final_answer = english_answer
            
            # Step 8: Generate speech if enabled
            audio_file = None
            if enable_tts and final_answer:
                audio_file = self.generate_speech(final_answer, lang_code)
            
            return final_answer, audio_file
            
        except Exception as e:
            logger.error(f"Answer processing error: {e}")
            error_msg = "Sorry, I encountered an error while processing your question. Please try again."
            if lang_code != "en":
                error_msg = self.translate_text(error_msg, target_lang=lang_code)
            return error_msg, None

# Initialize bot
bot = TouristGuideBot()

def process_text_input(question, language, enable_tts):
    """Process text input from user."""
    if not question.strip():
        return "Please enter a question.", None
    
    answer, audio = bot.answer_question(question, language, enable_tts)
    return answer, audio

def process_audio_input(audio, language, enable_tts):
    """Process audio input from user."""
    if audio is None:
        return "Please record an audio message.", None
    
    # Convert speech to text
    question = bot.process_audio_input(audio)
    if not question or question.startswith("Could not understand"):
        return question, None
    
    # Process the transcribed question
    answer, audio_response = bot.answer_question(question, language, enable_tts)
    return f"Your question: {question}\n\nAnswer: {answer}", audio_response

# Create Gradio interface
def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="🌍 Multilingual Tourist Guide Bot", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 🌍 Multilingual Tourist Guide Bot
        
        Your AI-powered travel companion that speaks your language! Get personalized travel advice, destination information, 
        and travel tips using RAG technology and live web data.
        
        **Features:**
        - 🗣️ Voice input and output
        - 🌐 Support for 10+ languages
        - 📚 Knowledge base + live web search
        - 🎯 Personalized travel recommendations
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                language_dropdown = gr.Dropdown(
                    choices=list(bot.languages.keys()),
                    value="English",
                    label="🌐 Select Language",
                    info="Choose your preferred language"
                )
                
                enable_tts_checkbox = gr.Checkbox(
                    value=True,
                    label="🔊 Enable Text-to-Speech",
                    info="Generate audio response"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### Ask your travel question in text or voice:")
        
        with gr.Tabs():
            with gr.TabItem("💬 Text Input"):
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(
                            label="Your Travel Question",
                            placeholder="Ask me anything about travel, destinations, tips, etc.",
                            lines=3
                        )
                        text_submit_btn = gr.Button("🚀 Get Answer", variant="primary")
                    
                    with gr.Column():
                        text_output = gr.Textbox(
                            label="Answer",
                            lines=8,
                            show_copy_button=True
                        )
                        text_audio_output = gr.Audio(label="🔊 Audio Response")
            
            with gr.TabItem("🎤 Voice Input"):
                with gr.Row():
                    with gr.Column():
                        audio_input = gr.Audio(
                            sources=["microphone"],
                            type="filepath",
                            label="🎤 Record Your Question"
                        )
                        audio_submit_btn = gr.Button("🚀 Process Audio", variant="primary")
                    
                    with gr.Column():
                        audio_text_output = gr.Textbox(
                            label="Answer",
                            lines=8,
                            show_copy_button=True
                        )
                        audio_audio_output = gr.Audio(label="🔊 Audio Response")
        
        # Example questions
        gr.Markdown("""
        ### 💡 Example Questions:
        - "What are the best places to visit in Paris?"
        - "Give me travel tips for first-time international travelers"
        - "What's the best time to visit Japan?"
        - "How can I save money while traveling in Europe?"
        - "Tell me about local customs in Dubai"
        """)
        
        # Event handlers
        text_submit_btn.click(
            fn=process_text_input,
            inputs=[text_input, language_dropdown, enable_tts_checkbox],
            outputs=[text_output, text_audio_output]
        )
        
        audio_submit_btn.click(
            fn=process_audio_input,
            inputs=[audio_input, language_dropdown, enable_tts_checkbox],
            outputs=[audio_text_output, audio_audio_output]
        )
        
        # Auto-submit on Enter for text input
        text_input.submit(
            fn=process_text_input,
            inputs=[text_input, language_dropdown, enable_tts_checkbox],
            outputs=[text_output, text_audio_output]
        )
    
    return demo

# Launch the app
if __name__ == "__main__":

    
    print("🚀 Starting Tourist Guide Bot...")
    print("📝 Note: Add your API keys as environment variables:")
    print("   - GROQ_API_KEY for Groq API")
    print("   - GEMINI_API_KEY for Google Gemini API")
    
    demo = create_interface()
    demo.launch(
        share=True,
        show_error=True
    )