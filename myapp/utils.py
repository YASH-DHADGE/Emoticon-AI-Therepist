# myapp/utils.py
import logging
import random
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime

logger = logging.getLogger(__name__)

# Global model instances
tokenizer = None
model = None

# Flag to use deepface
USE_DEEPFACE = True
try:
    from deepface import DeepFace
except ImportError:
    USE_DEEPFACE = False
    logger.warning("DeepFace import failed. Emotion detection will be disabled.")

# Emotional response templates
EMOTION_TEMPLATES = {
    'happy': [
        "I'm glad you're feeling positive! {response}",
        "That's wonderful to hear! {response}",
        "Your happiness is contagious! {response}"
    ],
    'sad': [
        "I understand this is difficult. {response}",
        "I'm here to support you through this. {response}",
        "It's okay to feel this way. {response}"
    ],
    'angry': [
        "I can sense your frustration. {response}",
        "Let's work through this together. {response}",
        "Your feelings are valid. {response}"
    ],
    'fear': [
        "I understand you're feeling anxious. {response}",
        "You're not alone in this. {response}",
        "Let's take it one step at a time. {response}"
    ],
    'neutral': [
        "{response}",
        "I hear you. {response}",
        "Thank you for sharing that. {response}"
    ]
}

# Crisis keywords and responses
CRISIS_KEYWORDS = [
    'suicide', 'kill myself', 'end it all', 'don\'t want to live',
    'self-harm', 'hurt myself', 'die', 'death'
]

CRISIS_RESPONSE = """
I'm very concerned about what you're telling me. While I'm here to listen, it's important that you get professional help.
Please consider these resources:
- National Suicide Prevention Lifeline (24/7): 988
- Crisis Text Line: Text HOME to 741741
- Emergency Services: 911
You're not alone in this, and there are people who want to help.
"""

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        logger.info("Loading DialoGPT model...")
        try:
            model_name = "microsoft/DialoGPT-medium"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    return tokenizer, model

def detect_emotion_safely(image_path: str) -> str:
    """Safely detect emotion from an image, returning 'neutral' if detection fails."""
    if not USE_DEEPFACE:
        return 'neutral'
    try:
        result = DeepFace.analyze(image_path, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion']
        return emotion
    except Exception as e:
        logger.error(f"Emotion detection failed: {e}")
        return 'neutral'

def check_crisis_keywords(message: str) -> bool:
    """Check if the message contains any crisis keywords."""
    return any(keyword in message.lower() for keyword in CRISIS_KEYWORDS)

def format_emotional_response(response: str, emotion: str) -> str:
    """Format the response based on the detected emotion."""
    templates = EMOTION_TEMPLATES.get(emotion, EMOTION_TEMPLATES['neutral'])
    return random.choice(templates).format(response=response)

def generate_response(history: List[Dict], mood: str = "", emotion_context: Optional[Dict] = None):
    try:
        # Check for crisis keywords in the latest message
        if history and check_crisis_keywords(history[-1]['content']):
            return CRISIS_RESPONSE
            
        # Model availability check
        if tokenizer is None or model is None:
            logger.error("Model not loaded at startup. Attempting to load now.")
            load_model()
            if tokenizer is None or model is None:
                raise Exception("Model is not available.")
        
        # Build context with emotional awareness
        chat_history_str = ""
        emotional_context = ""
        
        # Add emotional context if available
        if emotion_context:
            pattern = emotion_context.get('pattern', '')
            duration = emotion_context.get('duration', '')
            if pattern and duration:
                emotional_context = f"[Emotional context: {pattern} for {duration}]"

        # Format conversation history with emotional awareness
        for turn in history[-5:]:
            content = turn['content']
            role = turn.get('role', 'user')
            # Add emotional markers for better context
            if role == 'user' and 'emotion' in turn:
                content = f"[Emotion: {turn['emotion']}] {content}"
            chat_history_str += content + tokenizer.eos_token
            
        # Add current emotional context
        if emotional_context:
            chat_history_str += emotional_context + tokenizer.eos_token
            
        # Add current mood if available
        if mood and mood != 'neutral':
            chat_history_str += f"[Current mood: {mood}] "
            
        # Add therapeutic prompts based on emotion
        if mood in ['sad', 'angry', 'fear']:
            chat_history_str += "[Respond with empathy and support] "
        elif mood == 'happy':
            chat_history_str += "[Maintain positive engagement] "

        # Tokenize input with emotional context
        inputs = tokenizer(
            chat_history_str,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        input_ids = inputs.input_ids
        
        # Generate response with enhanced parameters for emotional support
        outputs = model.generate(
            input_ids,
            max_length=len(input_ids[0]) + 150,  # Longer responses for better emotional support
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.92,
            temperature=0.85,  # Slightly higher temperature for more emotionally nuanced responses
            repetition_penalty=1.2,  # Reduce repetition
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        
        # Decode and format response
        response = tokenizer.decode(outputs[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        # Format response based on emotion
        formatted_response = format_emotional_response(response, mood)
    
        return formatted_response
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        # Enhanced fallback responses with emotional awareness
        fallbacks = {
            'sad': [
                "I hear that you're going through a difficult time. Would you like to tell me more about what's troubling you?",
                "I'm here to listen and support you. Can you help me understand what you're feeling?",
                "It's okay to feel this way. Would you like to explore these feelings together?"
            ],
            'angry': [
                "I can sense your frustration. Would you like to talk about what's bothering you?",
                "Your feelings are valid. Can you tell me more about what's making you feel this way?",
                "I'm here to listen without judgment. What would help you feel better right now?"
            ],
            'fear': [
                "I understand you're feeling anxious. Let's take it one step at a time. What's on your mind?",
                "You're not alone in this. Would you like to talk about what's causing these feelings?",
                "I'm here to support you through this. What would help you feel safer right now?"
            ],
            'happy': [
                "That's wonderful! Would you like to share more about what's making you feel good?",
                "I'm glad you're feeling positive! What's contributing to your happiness?",
                "It's great to hear you're doing well! Would you like to explore what's going right?"
            ],
            'neutral': [
                "I'm here to listen and support you. Would you like to share what's on your mind?",
                "I'm interested in understanding your perspective better. Could you tell me more?",
                "Let's explore this together. What would you like to focus on?"
            ]
        }
        
        selected_fallbacks = fallbacks.get(mood, fallbacks['neutral'])
        return random.choice(selected_fallbacks)