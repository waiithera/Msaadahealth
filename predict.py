import json
import torch
from nnModel import NeuralNet
from stem import bagOfWords, tokenizeWords
import random
from typing import Tuple, Dict
import re
from datetime import datetime

class HealthBot:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Load intents and model
            with open('intents.json', 'r', encoding='utf-8') as file:
                self.intents = json.load(file)
            
            # Load trained model
            FILE = "data.pth"
            data = torch.load(FILE)
            
            self.all_words = data['all_words']
            self.tags = data['tags']
            model_state = data["model_state"]
            input_size = data["input_size"]
            hidden_size = data["hidden_size"]
            output_size = data["output_size"]
            
            # Initialize neural network
            self.model = NeuralNet(input_size, hidden_size, output_size).to(self.device)
            self.model.load_state_dict(model_state)
            self.model.eval()
            
            print("Model and intents loaded successfully")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def detect_language(self, text: str) -> str:
        """Detect language based on input text."""
        text = text.lower()
        
        lang_patterns = {
            'sw': ['habari', 'jambo', 'afya', 'sukari', 'dawa', 'chakula', 'tafadhali', 'asante'],
            'sheng': ['sasa', 'poa', 'fiti', 'rada', 'daro', 'omundu', 'nini', 'maze'],
            'en': ['hello', 'hi', 'sugar', 'diabetes', 'medicine', 'food', 'please', 'thank']
        }
        
        scores = {
            lang: sum(1 for word in patterns if word in text)
            for lang, patterns in lang_patterns.items()
        }
        
        return max(scores.items(), key=lambda x: x[1])[0] if any(scores.values()) else 'en'

    def get_bot_response(self, text: str, language: str = 'en') -> Tuple[str, str]:
        """Get response using neural network with language support."""
        try:
            # Neural network processing
            sentence = tokenizeWords(text)
            X = bagOfWords(sentence, self.all_words)
            X = X.reshape(1, X.shape[0])
            X = torch.from_numpy(X).to(self.device)
            
            output = self.model(X)
            _, predicted = torch.max(output, dim=1)
            
            tag = self.tags[predicted.item()]
            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]

            # If confidence is high enough, get response in appropriate language
            if prob.item() > 0.75:
                for intent in self.intents['intents']:
                    if tag == intent["tag"]:
                        # Try to get response in requested language, fall back to English
                        responses = intent.get(f'responses_{language}', intent.get('responses', []))
                        if not responses:
                            responses = intent.get('responses', [])
                        
                        if responses:
                            response = random.choice(responses)
                            return response, tag

            # Default responses in different languages
            defaults = {
                'en': "I'm not quite sure I understand. Could you rephrase that?",
                'sw': "Samahani, sijaelewa vizuri. Unaweza kurudia tafadhali?",
                'sheng': "Sijaget vizuri. Unaweza repeat hiyo?"
            }
            return defaults.get(language, defaults['en']), 'unknown'

        except Exception as e:
            print(f"Error in get_bot_response: {str(e)}")
            error_messages = {
                'en': "Sorry, I encountered an error. Please try again.",
                'sw': "Samahani, kuna hitilafu. Tafadhali jaribu tena.",
                'sheng': "Kuna issue. Please try tena."
            }
            return error_messages.get(language, error_messages['en']), 'error'

# Create global bot instance
bot = HealthBot()

def getBotInput(text: str, language: str = None) -> str:
    """Get chatbot response with language detection."""
    try:
        # Detect language if not specified
        if language is None:
            language = bot.detect_language(text)
        
        response, _ = bot.get_bot_response(text, language)
        return response
    except Exception as e:
        print(f"Error in getBotInput: {str(e)}")
        return "Sorry, I encountered an error processing your request."

def getTag(text: str) -> str:
    """Get status tag for message."""
    try:
        _, tag = bot.get_bot_response(text)
        
        if tag == "goodbye":
            return "Offline"
        return "Online"
    except Exception as e:
        print(f"Error in getTag: {str(e)}")
        return "Online"