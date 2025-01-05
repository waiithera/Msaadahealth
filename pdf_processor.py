"""
PDF Processor module for chatbot training data generation.
Extracts text from PDFs and converts it into question-answer pairs and intents.
"""

import os
import PyPDF2
import json
import re
import nltk
from typing import List, Dict, Any
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pathlib import Path

class PDFChatbotTrainer:
    def __init__(self, nltk_data_path: str = None, active: bool = False):
        self.active = active
        if nltk_data_path:
            nltk.data.path.append(nltk_data_path)
    
    def __init__(self, nltk_data_path: str = None):
        """
        Initialize the PDFChatbotTrainer.
        
        Args:
            nltk_data_path (str, optional): Custom path for NLTK data. 
                                          If None, uses default NLTK path.
        """
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
        
        # Define supported file types
        self.supported_extensions = {'.pdf'}

    def clean_data(self, text: str) -> str:
        """Clean text data."""
        import re
        text = re.sub(r'\ufb00', 'ff', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,?!-]', '', text)
        return text.strip()

    def validate_file(self, pdf_path: str) -> bool:
        """
        Validate if the file exists and is a PDF.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            bool: True if file is valid, False otherwise
            
        Raises:
            ValueError: If file is not a PDF or doesn't exist
        """
        path = Path(pdf_path)
        if not path.exists():
            raise ValueError(f"File not found: {pdf_path}")
        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported file type. Supported types: {self.supported_extensions}")
        return True

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
            
        Raises:
            PyPDF2.PdfReadError: If PDF is encrypted or corrupted
        """
        self.validate_file(pdf_path)
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
        except PyPDF2.PdfReadError as e:
            raise PyPDF2.PdfReadError(f"Error reading PDF: {str(e)}")
        return text

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess the extracted text"""
        # Remove special characters and extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,?]', '', text)
        return text.strip()
    
    

    def extract_qa_pairs(self, text: str) -> List[Dict[str, str]]:
        """
        Extract question-answer pairs from text.
        
        Args:
            text (str): Preprocessed text to extract QA pairs from
            
        Returns:
            List[Dict[str, str]]: List of dictionaries containing questions and answers
        """
        sentences = sent_tokenize(text)
        qa_pairs = []
        
        question_patterns = [
            r'What is.*\?',
            r'How (do|does|can).*\?',
            r'Why.*\?',
            r'When.*\?',
            r'Where.*\?',
            r'Is.*\?',
            r'Are.*\?',
            r'Can.*\?'
        ]

        for i in range(len(sentences)-1):
            current_sent = sentences[i]
            next_sent = sentences[i+1]
            
            is_question = any(re.match(pattern, current_sent, re.IGNORECASE) 
                            for pattern in question_patterns)
            
            if is_question:
                qa_pairs.append({
                    "question": current_sent.strip(),
                    "answer": next_sent.strip()
                })
            
            # Create statement-based patterns for relevant keywords
            for keyword in ["diabetes", "treatment", "symptoms", "diagnosis"]:
                if keyword in current_sent.lower():
                    statement = current_sent.strip()
                    if statement.endswith('.'):
                        statement = statement[:-1]
                    
                    qa_pairs.append({
                        "question": f"What can you tell me about {statement.lower()}?",
                        "answer": current_sent.strip()
                    })

        return qa_pairs

    def convert_to_intents(self, qa_pairs: List[Dict[str, str]]) -> Dict[str, List]:
        """
        Convert QA pairs to intents format.
        
        Args:
            qa_pairs (List[Dict[str, str]]): List of QA pairs
            
        Returns:
            Dict[str, List]: Intent data structure
        """
        diabetes_intents = {
            "intents": []
        }
        
        topics = {
            "symptoms": ["symptom", "sign", "feel", "feeling"],
            "treatment": ["treat", "medication", "medicine", "insulin", "drug"],
            "diagnosis": ["diagnose", "test", "check", "measure"],
            "prevention": ["prevent", "avoid", "reduce risk", "lifestyle"],
            "diet": ["food", "eat", "diet", "meal", "sugar", "carb"],
            "exercise":["workout", "walk", "run"],
            "complications": ["complication", "problem", "risk", "affect"],
            "general": []
        }

        for qa_pair in qa_pairs:
            topic_found = False
            for topic, keywords in topics.items():
                if any(keyword in qa_pair["question"].lower() for keyword in keywords):
                    tag = f"diabetes_{topic}"
                    topic_found = True
                    break
            
            if not topic_found:
                tag = "diabetes_general"

            # Check if intent exists
        intent_exists = False
        for intent in diabetes_intents["intents"]:
            if intent["tag"] == tag:
                intent["patterns"].append(self.clean_data(qa_pair["question"]))
                if self.clean_data(qa_pair["answer"]) not in intent["responses"]:
                    intent["responses"].append(self.clean_data(qa_pair["answer"]))
                intent_exists = True
                break

        if not intent_exists:
            diabetes_intents["intents"].append({
                "tag": tag,
                "patterns": [self.clean_data(qa_pair["question"])],
                "responses": [self.clean_data(qa_pair["answer"])]
            })


        return diabetes_intents

def process_pdf_to_intents(self, pdf_path: str, output_path: str) -> int:
    if not self.active:
        return 0