import os
import sys
import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from stem import bagOfWords, tokenizeWords, stemWords
from nnModel import NeuralNet
from pdf_processor import PDFChatbotTrainer
from web_processor import WebChatbotTrainer

# Check NumPy version compatibility
if np.__version__.startswith('2.'):
    print("Warning: NumPy 2.x detected. This script requires NumPy 1.x")
    print("Please run: pip install numpy==1.24.3")
    sys.exit(1)

def get_patterns(intent):
    """Combine patterns from all languages"""
    try:
        all_patterns = []
        for key in intent:
            if key.startswith('patterns_'):
                all_patterns.extend(intent[key])
        return all_patterns
    except Exception as e:
        print(f"Error in get_patterns: {e}")
        return []

class ChatDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.n_samples = len(x_data)
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

def train_model():
    try:
        # Load and process intents
        intents_path = 'D:/medical-assistant-chatbot-main (3)/medical-assistant-chatbot-main/intents.json'
        if not os.path.exists(intents_path):
            raise FileNotFoundError(f"Intents file not found at {intents_path}")
            
        with open(intents_path, 'r', encoding='utf-8') as f:
            intents = json.load(f)

        tags = []
        allWords = []
        xy = []

        for intent in intents['intents']:
            tag = intent['tag']
            tags.append(tag)
            
            patterns = get_patterns(intent)
            for pattern in patterns:
                rootWords = tokenizeWords(pattern)
                allWords.extend(rootWords)
                xy.append((rootWords, tag))

        # Data preprocessing
        ignoreWords = ['?', '.', '!']
        allWords = [stemWords(words) for words in allWords if words not in ignoreWords]
        allWords = sorted(set(allWords))
        tags = sorted(set(tags))

        if not allWords or not tags:
            raise ValueError("No words or tags found after preprocessing")

        # Create training data
        x_train = []
        y_train = []
        for (pattern_sentence, tag) in xy:
            bag = bagOfWords(pattern_sentence, allWords)
            x_train.append(bag)
            label = tags.index(tag)
            y_train.append(label)

        # Convert to numpy arrays with error handling
        try:
            x_train = np.array(x_train, dtype=np.float32)
            y_train = np.array(y_train, dtype=np.int64)
        except Exception as e:
            print(f"Error converting to NumPy arrays: {e}")
            raise

        # Convert to tensors with error handling
        try:
            x_train = torch.from_numpy(x_train)
            y_train = torch.from_numpy(y_train)
        except Exception as e:
            print(f"Error converting to PyTorch tensors: {e}")
            raise

        # Training parameters
        epochs = 6000
        learningRate = 0.001
        batchSize = 30
        inputSize = len(x_train[0])
        hiddenSize = 30
        outputSize = len(tags)

        # Training setup
        dataset = ChatDataset(x_train, y_train)
        train_loader = DataLoader(dataset=dataset, batch_size=batchSize, shuffle=True, num_workers=0)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = NeuralNet(inputSize, hiddenSize, outputSize).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

        print(f"Training on device: {device}")
        print(f"Input size: {inputSize}, Hidden size: {hiddenSize}, Output size: {outputSize}")

        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            for (words, labels) in train_loader:
                try:
                    words = words.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(words)
                    loss = criterion(outputs, labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                except RuntimeError as e:
                    print(f"Error during training iteration: {e}")
                    continue
                
            if (epoch+1) % 100 == 0:
                avg_loss = total_loss / len(train_loader)
                print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')

        print(f'Final loss: {loss.item():.4f}')

        # Save model
        data = {
            "model_state": model.state_dict(),
            "input_size": inputSize,
            "hidden_size": hiddenSize,
            "output_size": outputSize,
            "all_words": allWords,
            "tags": tags
        }

        FILE = "D:/medical-assistant-chatbot-main (3)/medical-assistant-chatbot-main/data.pth"
        torch.save(data, FILE)
        print(f'Training complete. File saved to {FILE}')

    except Exception as e:
        print(f"Error in train_model: {str(e)}")
        raise

def main():
    try:
        # Initialize trainers
        pdf_trainer = PDFChatbotTrainer(nltk_data_path="./nltk_data")
        web_trainer = WebChatbotTrainer()
        
        keywords = [
            "diabetes", "blood sugar", "meal plan", "diagnosis",
            "insulin", "glucose", "diet", "nutrition", "medicine",
            "mellitus", "HbA1c", "hypoglycemia", "hyperglycemia"
        ]
        
        topics = {
            "diabetes_general": ["diabetes", "mellitus"],
            "blood_sugar": ["blood sugar", "glucose", "HbA1c", "hypoglycemia", "hyperglycemia"],
            "treatment": ["insulin", "medicine"],
            "diet": ["meal plan", "diet", "nutrition"],
            "diagnosis": ["diagnosis"],
            "general": []
        }

        # Process PDF
        try:
            num_intents = pdf_trainer.process_pdf_to_intents(
                pdf_path=r"D:/medical-assistant-chatbot-main (3)/medical-assistant-chatbot-main/data/diabetes2.pdf",
                output_path=r"D:/medical-assistant-chatbot-main (3)/medical-assistant-chatbot-main/intents.json"
            )
            print(f"Successfully processed PDF and generated {num_intents} intents")
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")

        # Process URLs
        urls = [
            "https://agamatrix.com/blog/diabetes-nutrition-myths/",
            "https://diabetesaction.org/questions-general-information",
            "https://www.eatingwell.com/article/290863/top-common-questions-about-diabetes/",
            "https://www.everydayhealth.com/type-1-diabetes/guide/"
        ]
        
        for url in urls:
            try:
                num_intents = web_trainer.process_url_to_intents(
                    url=url,
                    output_path=r"D:/medical-assistant-chatbot-main (3)/medical-assistant-chatbot-main/intents.json",
                    keywords=keywords,
                    topics=topics
                )
                print(f"Processed {url}: Generated {num_intents} intents")
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
                continue

        # Print final summary
        try:
            web_trainer.print_intents_summary(r"D:/medical-assistant-chatbot-main (3)/medical-assistant-chatbot-main/intents.json")
        except Exception as e:
            print(f"Error printing summary: {str(e)}")

        # Start training
        train_model()

    except Exception as e:
        print(f"Error in main function: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()