"""
Web Processor module for chatbot training data generation.
Extracts text from websites and converts it into question-answer pairs and intents.
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import nltk
from typing import List, Dict, Any
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from urllib.parse import urlparse
import os

class WebChatbotTrainer:
    def __init__(self, nltk_data_path: str = None, active: bool = False):
        self.active = active
        if nltk_data_path:
            nltk.data.path.append(nltk_data_path)
    
    def __init__(self, nltk_data_path: str = None):
        """Initialize the WebChatbotTrainer"""
        # Set custom NLTK data path if provided
        if nltk_data_path:
            nltk.data.path.append(nltk_data_path)
        
        # Ensure NLTK data is downloaded
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))

    def validate_url(self, url: str) -> bool:
        """
        Validate if the URL is properly formatted and accessible.
        
        Args:
            url (str): URL to validate
            
        Returns:
            bool: True if URL is valid and accessible
            
        Raises:
            ValueError: If URL is invalid or inaccessible
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            raise ValueError(f"Invalid URL format: {url}")

    def extract_text_from_url(self, url: str) -> str:
        """
        Extract text content from a website URL.
        
        Args:
            url (str): Website URL to extract text from
            
        Returns:
            str: Extracted text from the website
            
        Raises:
            requests.RequestException: If there's an error accessing the URL
        """
        self.validate_url(url)
        
        try:
            # Send request with headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "header", "footer", "nav"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except requests.RequestException as e:
            raise requests.RequestException(f"Error accessing URL: {str(e)}")

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess the extracted text"""
        # Remove special characters and extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,?]', '', text)
        return text.strip()

    def extract_qa_pairs(self, text: str, keywords: List[str]) -> List[Dict[str, str]]:
        """
        Extract question-answer pairs from text.
        
        Args:
            text (str): Preprocessed text to extract QA pairs from
            keywords (List[str]): List of keywords to look for in the text
            
        Returns:
            List[Dict[str, str]]: List of dictionaries containing questions and answers
        """
        sentences = sent_tokenize(text)
        qa_pairs = []
        
        # Common question patterns
        question_patterns = [
            r'What is.*\?',
            r'How (do|does|can).*\?',
            r'Why.*\?',
            r'When.*\?',
            r'Where.*\?',
            r'Is.*\?',
            r'Are.*\?',
            r'Can.*\?',
            r'Should.*\?',
            r'Who.*\?',
            r'What.*\?',
            r'Which.*\?',
            r'Whom.*\? '
        ]

        for i in range(len(sentences)-1):
            current_sent = sentences[i]
            next_sent = sentences[i+1]
            
            # Check for explicit questions
            is_question = any(re.match(pattern, current_sent, re.IGNORECASE) 
                            for pattern in question_patterns)
            
            if is_question:
                qa_pairs.append({
                    "question": current_sent.strip(),
                    "answer": next_sent.strip()
                })
            
            # Create statement-based patterns for relevant keywords
            for keyword in keywords:
                if keyword.lower() in current_sent.lower():
                    statement = current_sent.strip()
                    if statement.endswith('.'):
                        statement = statement[:-1]
                    
                    qa_pairs.append({
                        "question": f"What can you tell me about {statement.lower()}?",
                        "answer": current_sent.strip()
                    })

        return qa_pairs

    def convert_to_intents(self, qa_pairs: List[Dict[str, str]], topics: Dict[str, List[str]]) -> Dict[str, List]:
        """
        Convert QA pairs to intents format.
        
        Args:
            qa_pairs (List[Dict[str, str]]): List of QA pairs
            topics (Dict[str, List[str]]): Dictionary of topics and their keywords
            
        Returns:
            Dict[str, List]: Intent data structure
        """
        intents_data = {
            "intents": []
        }

        for qa_pair in qa_pairs:
            topic_found = False
            for topic, keywords in topics.items():
                if any(keyword in qa_pair["question"].lower() for keyword in keywords):
                    tag = f"{topic}"
                    topic_found = True
                    break
            
            if not topic_found:
                tag = "general"

            # Check if intent exists
            intent_exists = False
            for intent in intents_data["intents"]:
                if intent["tag"] == tag:
                    intent["patterns"].append(qa_pair["question"])
                    if qa_pair["answer"] not in intent["responses"]:
                        intent["responses"].append(qa_pair["answer"])
                    intent_exists = True
                    break

           # if not intent_exists:
                intents_data["intents"].append({
                    "tag": tag,
                    "patterns": [qa_pair["question"]],
                    "responses": [qa_pair["answer"]]
                })

        return intents_data  
    

def process_url_to_intents(self, url: str, output_path: str, keywords: List[str], topics: Dict[str, List[str]]) -> int:
    if not self.active:
        return 0

    def print_intents_summary(self, output_path: str):
        """Print a summary of all generated intents"""
        try:
            with open(output_path, 'r') as f:
                intents_data = json.load(f)
                
            print("\n=== Intents Summary ===")
            for intent in intents_data["intents"]:
                print(f"\nTag: {intent['tag']}")
                print(f"Number of patterns: {len(intent['patterns'])}")
                print("Example patterns:")
                for pattern in intent['patterns'][:3]:  # Show first 3 patterns
                    print(f"  - {pattern}")
                print(f"Number of responses: {len(intent['responses'])}")
                
            print(f"\nTotal number of intents: {len(intents_data['intents'])}")
            
        except FileNotFoundError:
            print(f"No intents file found at {output_path}")
        except json.JSONDecodeError:
            print("Error reading intents file: Invalid JSON format")