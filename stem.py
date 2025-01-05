import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
import re
import string
from typing import List

# Initialize Porter Stemmer
stemming = PorterStemmer()

def clean_text(text: str) -> str:
    """Clean and normalize text data."""
    # Convert to lowercase
    text = text.lower().strip()
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def tokenizeWords(sentence: str) -> List[str]:
    """Tokenize input sentence after cleaning."""
    # Clean the text first
    cleaned_text = clean_text(sentence)
    # Use NLTK tokenizer
    return nltk.word_tokenize(cleaned_text)

def stemWords(word: str) -> str:
    """Find the root form of the word."""
    # Clean the word first
    cleaned_word = clean_text(word)
    # Apply stemming
    return stemming.stem(cleaned_word.lower())

def bagOfWords(tokenizedSentence: List[str], words: List[str]) -> np.ndarray:
    """Create bag of words array from tokenized sentence."""
    # Stem each word in the input sentence
    wordsInSentence = [stemWords(element) for element in tokenizedSentence]
    
    # Initialize bag with zeros
    bagOfWords = np.zeros(len(words), dtype=np.float32)
    
    # Fill the bag
    for idex, words1 in enumerate(words):
        # Stem the comparison word
        stemmed_word = stemWords(words1)
        if stemmed_word in wordsInSentence:
            bagOfWords[idex] = 1

    return bagOfWords

# Download required NLTK data (only needs to be done once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')