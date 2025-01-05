import os
import bcrypt
from flask import Flask, flash, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from flask_sqlalchemy import SQLAlchemy
import mysql
from werkzeug.security import generate_password_hash, check_password_hash
import torch
import json
from datetime import datetime
from nnModel import NeuralNet
from stem import bagOfWords, tokenizeWords
import random
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import click
from flask.cli import with_appcontext
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SECRET_KEY'] = 'your_secret_key_here'  # Change this to a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{os.getenv('root')}:{os.getenv('')}@{os.getenv('localhost')}/{os.getenv('msaadahealth')}"
app.config['SQLALCHEMY_POOL_SIZE'] = 10
app.config['SQLALCHEMY_MAX_OVERFLOW'] = 20
app.config['SQLALCHEMY_POOL_TIMEOUT'] = 30


os.makedirs(os.path.join(basedir, 'instance'), exist_ok=True)


CORS(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Language Configuration
SUPPORTED_LANGUAGES = {
    'en': {
        'name': 'English',
        'messages': {
            'welcome': 'Welcome to Chatbot',
            'login_required': 'Please login to continue',
            'invalid_credentials': 'Invalid username or password',
            'registration_success': 'Registration successful',
            'logout_success': 'Successfully logged out',
            'error': 'An error occurred',
            'send_message': 'Send a message...',
            'new_chat': 'New Chat',
            'search_placeholder': 'Search chats...',
            'no_chats': 'No chats.',
            'tools': 'Tools',
            'files': 'Files',
            'settings': 'Settings',
            'quick_settings': 'Quick Settings'
        }
    },
    'sw': {
        'name': 'Swahili',
        'messages': {
            'welcome': 'Karibu kwenye Chatbot',
            'login_required': 'Tafadhali ingia kwanza',
            'invalid_credentials': 'Jina la mtumiaji au nywila si sahihi',
            'registration_success': 'Usajili umefanikiwa',
            'logout_success': 'Umetoka kikamilifu',
            'error': 'Hitilafu imetokea',
            'send_message': 'Tuma ujumbe...',
            'new_chat': 'Mazungumzo Mapya',
            'search_placeholder': 'Tafuta mazungumzo...',
            'no_chats': 'Hakuna mazungumzo.',
            'tools': 'Vifaa',
            'files': 'Faili',
            'settings': 'Mipangilio',
            'quick_settings': 'Mipangilio ya Haraka'
        }
    },
    'sheng': {
        'name': 'Sheng',
        'messages': {
            'welcome': 'Karibu Chatbot',
            'login_required': 'Login kwanza fam',
            'invalid_credentials': 'Username ama password si poa',
            'registration_success': 'Registration imekam poa',
            'logout_success': 'Umetoka poa',
            'error': 'Kuna noma fulani',
            'send_message': 'Tuma message...',
            'new_chat': 'Chat Mpya',
            'search_placeholder': 'Search ma chat...',
            'no_chats': 'Hakuna ma chat.',
            'tools': 'Tools',
            'files': 'Ma File',
            'settings': 'Settings',
            'quick_settings': 'Quick Settings'
        }
    }
}

# Database Models
class User(db.Model):
    __tablename__ = 'user'  # Specify the exact table name
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))  # Changed to match your column name
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    messages = db.relationship('Message', backref='chat', lazy=True)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    is_bot = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    language = db.Column(db.String(10), default='en')
    chat_id = db.Column(db.Integer, db.ForeignKey('chat.id'), nullable=False)

# Load the chatbot model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    with open('intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)
except FileNotFoundError:
    print("Error: intents.json file not found")
    intents = {"intents": []}

try:
    data = torch.load("data.pth")
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data["all_words"]
    tags = data["tags"]
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
except FileNotFoundError:
    print("Error: data.pth file not found")
    model = None

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))



def get_response(message, lang='en'):
    if model is None:
        return "Model not loaded properly. Please check the model file."

    sentence = tokenizeWords(message)
    X = bagOfWords(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                responses_key = f"responses_{lang}"
                default_responses_key = "responses_en"
                responses = intent.get(responses_key, intent.get(default_responses_key, []))
                if responses:
                    return random.choice(responses)
    
    # Default responses in different languages
    default_responses = {
        'en': "I'm not sure I understand. Could you rephrase that?",
        'sw': "Sifahamu vizuri. Unaweza kusema tena?",
        'sheng': "Sijapata vizuri. Unaweza repeat?"
    }
    
    return default_responses.get(lang, default_responses['en'])

# Routes
@app.route('/')
def index():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    return redirect(url_for('chat_interface'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('Username')
        password = request.form.get('Password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('chat'))
        else:
            flash('Invalid username or password')
            
    return render_template('login.html')

def init_db():
    with app.app_context():
        try:
            db.create_all()
            print("Database tables created successfully!")
            return True
        except Exception as e:
            print(f"Error creating database tables: {e}")
            return False


@app.route('/debug/users')
def debug_users():
    if app.debug:  # Only available in debug mode
        try:
            users = User.query.all()
            return {
                'user_count': len(users),
                'users': [
                    {
                        'id': user.id,
                        'username': user.username,
                        'email': user.email
                    } for user in users
                ]
            }
        except Exception as e:
            return {'error': str(e)}
    return 'Debugging not available'

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            username = request.form.get('Username')
            password = request.form.get('Password')
            email = request.form.get('Email')
            
            if not all([username, password, email]):
                flash('All fields are required')
                return render_template('register.html')
            
            # Check for existing user
            if User.query.filter_by(username=username).first():
                flash('Username already exists')
                return render_template('register.html')
                
            if User.query.filter_by(email=email).first():
                flash('Email already registered')
                return render_template('register.html')
            
            # Create new user
            new_user = User(username=username, email=email)
            new_user.set_password(password)
            
            db.session.add(new_user)
            db.session.commit()
            
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
            
        except Exception as e:
            print(f"Registration error: {str(e)}")
            db.session.rollback()
            flash('Error occurred during registration')
            return render_template('register.html')
    
    return render_template('register.html')

        
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/chat-interface')
@login_required
def chat_interface():
    lang = session.get('language', 'en')
    return render_template('index.html', 
                         languages=SUPPORTED_LANGUAGES,
                         current_language=lang)

@app.route('/change-language', methods=['POST'])
def change_language():
    try:
        data = request.get_json()
        lang = data.get('language', 'en')
        
        if lang not in SUPPORTED_LANGUAGES:
            return jsonify({
                "status": "error",
                "message": "Unsupported language"
            }), 400
            
        session['language'] = lang
        
        return jsonify({
            "status": "success",
            "language": lang,
            "messages": SUPPORTED_LANGUAGES[lang]['messages']
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/get-current-language', methods=['GET'])
def get_current_language():
    lang = session.get('language', 'en')
    return jsonify({
        "language": lang,
        "messages": SUPPORTED_LANGUAGES[lang]['messages']
    })

@app.route('/create-chat', methods=['POST'])
@login_required
def create_chat():
    chat = Chat(user_id=current_user.id, title="New Chat")
    db.session.add(chat)
    db.session.commit()
    return jsonify({
        "status": "success",
        "chat_id": chat.id,
        "title": chat.title
    })

@app.route('/get-chats', methods=['GET'])
@login_required
def get_chats():
    chats = Chat.query.filter_by(user_id=current_user.id).order_by(Chat.created_at.desc()).all()
    return jsonify({
        "chats": [{
            "id": chat.id,
            "title": chat.title,
            "created_at": chat.created_at.isoformat()
        } for chat in chats]
    })

@app.route('/get-messages/<int:chat_id>', methods=['GET'])
@login_required
def get_messages(chat_id):
    chat = Chat.query.get_or_404(chat_id)
    if chat.user_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403
    
    messages = Message.query.filter_by(chat_id=chat_id).order_by(Message.timestamp).all()
    return jsonify({
        "messages": [{
            "id": msg.id,
            "content": msg.content,
            "is_bot": msg.is_bot,
            "timestamp": msg.timestamp.isoformat(),
            "language": msg.language
        } for msg in messages]
    })

@app.route('/send-message', methods=['POST'])
@login_required
def send_message():
    try:
        data = request.get_json()
        message_content = data.get('message', '').strip()
        chat_id = data.get('chat_id')
        lang = data.get('language', session.get('language', 'en'))

        if not message_content or not chat_id:
            return jsonify({"error": "Message or chat_id missing"}), 400

        # Save user message
        user_message = Message(
            content=message_content,
            is_bot=False,
            chat_id=chat_id,
            language=lang
        )
        db.session.add(user_message)

        # Get bot response
        bot_response = get_response(message_content, lang)

        # Save bot message
        bot_message = Message(
            content=bot_response,
            is_bot=True,
            chat_id=chat_id,
            language=lang
        )
        db.session.add(bot_message)
        db.session.commit()

        return jsonify({
            "status": "success",
            "response": bot_response,
            "message_id": user_message.id
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.context_processor
def inject_languages():
    return {
        'languages': SUPPORTED_LANGUAGES,
        'current_language': session.get('language', 'en')
    }

if __name__ == '__main__':
    if init_db():
        app.run(debug=True)
    else:
        print("Failed to initialize database. Please check your configuration.")