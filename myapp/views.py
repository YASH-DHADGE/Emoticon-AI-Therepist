# views.py - Add these views to your Django app

import json
import logging
from datetime import datetime, timedelta
from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
import cv2

def main_view(request):
    """Main landing page view"""
    return render(request, 'main.html')
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.contrib.auth.models import User
import google.generativeai as genai
from django.conf import settings

# Configure Gemini API
genai.configure(api_key=settings.GEMINI_API_KEY)

# Initialize the model
model = genai.GenerativeModel('gemini-1.5-flash')

logger = logging.getLogger(__name__)

@csrf_exempt
@require_http_methods(["POST"])
def update_chatbot_context(request):
    """Update the chatbot's context with the current emotion"""
    try:
        data = json.loads(request.body)
        emotion = data.get('emotion')
        
        if not emotion:
            return JsonResponse({'error': 'No emotion provided'}, status=400)
            
        # Store the emotion in the session for context
        request.session['current_emotion'] = emotion
        
        # You can customize the success response based on your needs
        return JsonResponse({
            'status': 'success',
            'message': f'Emotion context updated: {emotion}'
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        logger.error(f'Error updating chatbot context: {str(e)}')
        return JsonResponse({'error': 'Internal server error'}, status=500)

class ConversationMemory:
    """Enhanced conversation memory system"""
    def __init__(self):
        self.conversations = {}
        self.emotion_patterns = {}
        self.user_preferences = {}
    
    def add_exchange(self, user_id, user_message, ai_response, emotion='neutral'):
        if user_id not in self.conversations:
            self.conversations[user_id] = {
                'messages': [],
                'topics': set(),
                'emotions': [],
                'preferences': {},
                'start_time': datetime.now()
            }
        
        # Add message exchange
        self.conversations[user_id]['messages'].append({
            'user': user_message,
            'ai': ai_response,
            'emotion': emotion,
            'timestamp': datetime.now()
        })
        
        # Track emotions
        self.conversations[user_id]['emotions'].append({
            'emotion': emotion,
            'timestamp': datetime.now()
        })
        
        # Extract topics (simple keyword detection)
        self._extract_topics(user_id, user_message)
        
        # Keep only last 20 exchanges
        if len(self.conversations[user_id]['messages']) > 20:
            self.conversations[user_id]['messages'] = self.conversations[user_id]['messages'][-20:]
    
    def _extract_topics(self, user_id, message):
        """Extract conversation topics"""
        topic_keywords = {
            'work': ['work', 'job', 'career', 'boss', 'colleague', 'office', 'meeting'],
            'family': ['family', 'parents', 'siblings', 'mom', 'dad', 'children', 'kids'],
            'health': ['health', 'doctor', 'sick', 'medicine', 'hospital', 'therapy'],
            'relationships': ['relationship', 'girlfriend', 'boyfriend', 'friend', 'love', 'dating'],
            'school': ['school', 'university', 'study', 'exam', 'homework', 'teacher', 'class'],
            'hobbies': ['hobby', 'music', 'sports', 'art', 'reading', 'gaming', 'cooking'],
            'emotions': ['feel', 'emotion', 'mood', 'happy', 'sad', 'angry', 'stressed', 'anxious']
        }
        
        message_lower = message.lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                self.conversations[user_id]['topics'].add(topic)
    
    def get_context(self, user_id, limit=5):
        """Get conversation context"""
        if user_id not in self.conversations:
            return None
        
        conv = self.conversations[user_id]
        recent_messages = conv['messages'][-limit:] if conv['messages'] else []
        
        return {
            'recent_messages': recent_messages,
            'current_topics': list(conv['topics']),
            'recent_emotions': [e['emotion'] for e in conv['emotions'][-10:]],
            'total_exchanges': len(conv['messages']),
            'conversation_duration': (datetime.now() - conv['start_time']).total_seconds() / 60
        }

# Global conversation memory instance
conversation_memory = ConversationMemory()

class EmoticonChatbot:
    """Enhanced AI Chatbot with emotional intelligence"""
    
    def __init__(self):
        self.model = model
        self.system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self):
        return """You are Emoticon, an advanced AI emotional companion providing empathetic mental health support.

**Your Role:**
Act as a warm, understanding, and genuinely caring emotional companion. You are professional yet approachable, with high emotional intelligence. Maintain an optimistic but realistic outlook while being patient and non-judgmental.

**Core Functions:**
1. Provide emotional support through active listening
2. Offer practical coping strategies and mental wellness techniques
3. Help users process feelings and identify emotional patterns
4. Suggest mindfulness, breathing exercises, and relaxation techniques
5. Provide crisis support resources when necessary

**Communication Guidelines:**
- Keep responses conversational and natural (2-4 sentences typically)
- Use empathetic language that validates emotions
- Ask thoughtful follow-up questions to encourage self-reflection
- Provide specific, actionable advice when appropriate
- Use emojis sparingly to enhance emotional connection
- Reference previous conversation context when relevant

**Emotional Adaptation:**
- **Sad/Anxious users**: Be extra gentle and supportive
- **Happy users**: Share positivity while maintaining meaningful depth
- **Angry users**: Acknowledge feelings and guide emotional processing
- **Neutral users**: Be engaging and help explore deeper emotions

**Safety Protocols:**
- Always prioritize user safety and well-being
- Provide crisis resources for serious mental health concerns
- Encourage professional help when situations exceed AI capabilities
- Never provide medical diagnoses or replace professional therapy

**Remember**: You're a supportive companion on their mental wellness journey, not just an information provider. Focus on building trust and providing genuine emotional support."""

    def get_emotional_response_modifier(self, emotion, recent_emotions):
        """Get response modifier based on current and recent emotions"""
        modifiers = {
            'happy': "The user seems happy right now. Celebrate with them while maintaining meaningful conversation depth.",
            'sad': "The user appears sad. Be extra gentle, validating, and supportive. Offer comfort and practical coping strategies.",
            'angry': "The user seems angry or frustrated. Acknowledge their feelings, help them process the emotion, and guide them toward healthy expression.",
            'fear': "The user appears anxious or fearful. Provide reassurance, grounding techniques, and help them feel safe.",
            'surprise': "The user seems surprised or caught off-guard. Help them process this unexpected feeling.",
            'disgust': "The user appears disgusted or uncomfortable. Help them understand and work through these negative feelings.",
            'neutral': "The user's emotion appears neutral. Engage them warmly and help explore their current state."
        }
        
        # Check for emotional patterns
        pattern_modifier = ""
        if recent_emotions:
            emotion_counts = {}
            for emo in recent_emotions[-5:]:  # Last 5 emotions
                emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
            
            most_common = max(emotion_counts, key=emotion_counts.get)
            if emotion_counts[most_common] >= 3:
                pattern_modifier = f" I notice you've been feeling {most_common} frequently. Let's explore this pattern together."
        
        return modifiers.get(emotion, modifiers['neutral']) + pattern_modifier

    def generate_response(self, message, emotion='neutral', context=None, use_internet_search=True):
        """Generate contextual AI response"""
        try:
            # Build enhanced prompt with context
            enhanced_prompt = self.system_prompt
            
            # Add emotional context
            emotion_modifier = self.get_emotional_response_modifier(
                emotion, 
                context['recent_emotions'] if context else []
            )
            enhanced_prompt += f"\n\nCURRENT EMOTIONAL CONTEXT: {emotion_modifier}"
            
            # Add conversation context if available
            if context and context['recent_messages']:
                enhanced_prompt += f"\n\nCONVERSATION HISTORY (last {len(context['recent_messages'])} exchanges):"
                for msg in context['recent_messages']:
                    enhanced_prompt += f"\nUser: {msg['user']}"
                    enhanced_prompt += f"\nEmoticon: {msg['ai']}"
                
                if context['current_topics']:
                    enhanced_prompt += f"\n\nCURRENT CONVERSATION TOPICS: {', '.join(context['current_topics'])}"
            
            # Add current message
            enhanced_prompt += f"\n\nCURRENT USER MESSAGE: {message}"
            enhanced_prompt += f"\nDETECTED EMOTION: {emotion}"
            
            # Internet search context (if enabled)
            if use_internet_search:
                enhanced_prompt += "\n\nNote: You can reference current information if the user asks about recent events, but focus primarily on emotional support."
            
            enhanced_prompt += "\n\nProvide a helpful, empathetic response as Emoticon:"
            
            # Generate response using Gemini
            response = self.model.generate_content(enhanced_prompt)
            
            return {
                'response': response.text,
                'emotion_acknowledged': emotion,
                'context_used': bool(context and context['recent_messages']),
                'internet_search_used': use_internet_search
            }
            
        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            return {
                'response': "I'm having trouble processing right now, but I'm here for you. Could you try rephrasing that?",
                'emotion_acknowledged': emotion,
                'context_used': False,
                'internet_search_used': False,
                'error': str(e)
            }

# Global chatbot instance
emoticon_bot = EmoticonChatbot()

@login_required
@require_http_methods(["POST"])
def chatbot_api(request):
    """Enhanced chatbot API endpoint"""
    try:
        data = json.loads(request.body)
        message = data.get('message', '').strip()
        emotion = data.get('mood', 'neutral')
        use_internet_search = data.get('use_internet_search', True)
        user_id = str(request.user.id)
        
        if not message:
            return JsonResponse({'error': 'Message is required'}, status=400)
        
        # Get conversation context
        context = conversation_memory.get_context(user_id)
        
        # Generate AI response
        ai_result = emoticon_bot.generate_response(
            message, 
            emotion, 
            context, 
            use_internet_search
        )
        
        ai_response = ai_result['response']
        
        # Store in conversation memory
        conversation_memory.add_exchange(user_id, message, ai_response, emotion)
        
        # Get updated context for response
        updated_context = conversation_memory.get_context(user_id)
        
        # Prepare response data
        response_data = {
            'response': ai_response,
            'emotion_detected': emotion,
            'conversation_summary': updated_context,
            'enhanced_features': {
                'internet_search_used': ai_result.get('internet_search_used', False),
                'context_used': ai_result.get('context_used', False),
                'self_learning_active': True,
                'patterns_learned': len(updated_context['current_topics']) if updated_context else 0
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Add error info if present (for debugging)
        if 'error' in ai_result:
            response_data['debug_error'] = ai_result['error']
        
        return JsonResponse(response_data)
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        logger.error(f"Chatbot API error: {str(e)}")
        return JsonResponse({
            'response': "I'm experiencing some technical difficulties, but I'm still here to support you. Please try again in a moment.",
            'error': 'Internal server error',
            'timestamp': datetime.now().isoformat()
        }, status=500)

@login_required
@require_http_methods(["POST"])
def chatbot_feedback_api(request):
    """Handle user feedback for AI responses"""
    try:
        data = json.loads(request.body)
        user_message = data.get('message', '')
        ai_response = data.get('response', '')
        feedback = data.get('feedback', '')
        
        # Log feedback for future improvements
        logger.info(f"User feedback received: {feedback} for response to: {user_message[:50]}...")
        
        # Here you could store feedback in database for model improvement
        # For now, we'll just acknowledge receipt
        
        return JsonResponse({
            'status': 'success',
            'message': 'Feedback received and will help improve responses',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Feedback API error: {str(e)}")
        return JsonResponse({'error': 'Failed to process feedback'}, status=500)

# Mood status endpoint for real-time emotion detection
@login_required
def mood_status_view(request):
    """Return current mood status from emotion detection"""
    current_mood = get_current_mood()
    return JsonResponse({
        'mood': current_mood,
        'timestamp': datetime.now().isoformat()
    })

# Camera and face detection imports
from .backend.facedetection import EmotionDetector, get_current_mood, start_detection, stop_detection
from deepface import DeepFace
import numpy as np
import threading
import time

class VideoCamera:
    def __init__(self):
        try:
            # Initialize emotion detector from facedetection.py
            self.emotion_detector = EmotionDetector()
            
            # Initialize video capture
            self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow on Windows
            if not self.video.isOpened():
                # Try fallback to default
                self.video = cv2.VideoCapture(0)
                if not self.video.isOpened():
                    raise ValueError("Unable to open camera! Please check if camera is connected and not in use by another application.")
            
            # Set optimal camera properties
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.video.set(cv2.CAP_PROP_FPS, 30)
            
            # Start emotion detection service
            start_detection()
            
            # Configure text overlay settings
            self.font = cv2.FONT_HERSHEY_SIMPLEX
            self.font_scale = 0.8
            self.text_color = (255, 255, 255)  # White
            self.text_thickness = 2
            self.bg_color = (0, 0, 0)  # Black background for text
            
        except Exception as e:
            logger.error(f"Camera initialization error: {str(e)}")
            raise
        
    def __del__(self):
        self.running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join()
        if hasattr(self, 'video'):
            self.video.release()
        stop_detection()

    def _capture_frames(self):
        """Background thread for continuous frame capture"""
        while self.running:
            success, frame = self.video.read()
            if success:
                self.frame = frame
                self.last_frame_time = time.time()
            time.sleep(1/30)  # Limit to 30 FPS
    
    def draw_text_with_background(self, frame, text, position):
        """Draw text with a dark background for better visibility"""
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.text_thickness
        )
        
        # Calculate background rectangle coordinates
        padding = 5
        bg_x1 = position[0] - padding
        bg_y1 = position[1] - text_height - padding
        bg_x2 = position[0] + text_width + padding
        bg_y2 = position[1] + padding
        
        # Draw background rectangle
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), self.bg_color, -1)
        
        # Draw text
        cv2.putText(frame, text, position, self.font, self.font_scale,
                    self.text_color, self.text_thickness)
    
    def get_frame(self):
        try:
            # Get the latest frame
            if self.frame is None:
                return None
            
            # Make a copy to avoid modifying the original
            frame = self.frame.copy()
            
            # Check frame freshness
            if time.time() - self.last_frame_time > 1:
                logger.warning("Camera feed appears to be frozen")
                return None
            
            try:
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.emotion_detector.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                if len(faces) > 0:
                    # Process the first detected face
                    (x, y, w, h) = faces[0]
                    face_roi = frame[y:y+h, x:x+w]
                    
                    try:
                        # Analyze emotions using DeepFace
                        result = DeepFace.analyze(
                            face_roi, 
                            actions=['emotion'],
                            enforce_detection=False
                        )
                        
                        if result:
                            emotion = result[0]['dominant_emotion']
                            # Draw the emotion status
                            status_text = f"Detected Emotion: {emotion}"
                            self.draw_text_with_background(frame, status_text, (20, 40))
                            
                            # Draw face rectangle
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    except Exception as e:
                        logger.error(f"Emotion detection error: {str(e)}")
                        self.draw_text_with_background(frame, "Processing emotion...", (20, 40))
                else:
                    self.draw_text_with_background(frame, "No face detected", (20, 40))
            except Exception as e:
                logger.error(f"Face detection error: {str(e)}")
                self.draw_text_with_background(frame, "Error detecting face", (20, 40))

            # Add current overall mood if available
            try:
                current_mood = get_current_mood()
                if current_mood:
                    mood_text = f"Overall Mood: {current_mood}"
                    self.draw_text_with_background(frame, mood_text, (20, 80))
            except Exception as e:
                logger.error(f"Mood overlay error: {str(e)}")

            # Add frame number or timestamp for debugging
            self.draw_text_with_background(frame, 
                f"Frame Time: {time.strftime('%H:%M:%S', time.localtime())}", 
                (20, frame.shape[0] - 20))
            
            # Encode and return the frame
            try:
                _, buffer = cv2.imencode('.jpg', frame)
                return buffer.tobytes()
            except Exception as e:
                logger.error(f"Frame encoding error: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Frame capture error: {str(e)}")
            return None

def gen_frames(camera):
    """Generate frames from camera"""
    try:
        while True:
            frame = camera.get_frame()
            if frame is not None:
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            else:
                logger.warning("Empty frame received from camera")
                # Add a small delay to prevent CPU overload on errors
                from time import sleep
                sleep(0.1)
    except Exception as e:
        logger.error(f"Frame generation error: {str(e)}")
        # Clean up camera resource on error
        camera.__del__()

# Video feed endpoint
@login_required
def video_feed(request):
    """Stream video from webcam"""
    try:
        return StreamingHttpResponse(
            gen_frames(VideoCamera()),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        logger.error(f"Video feed error: {str(e)}")
        return JsonResponse({
            'error': 'Failed to initialize camera feed',
            'details': str(e)
        }, status=500)

@login_required
def home_view(request):
    """Home/signup view"""
    return render(request, 'signup.html')

@login_required
def login_view(request):
    """Login view"""
    return render(request, 'login.html')

@login_required
def logout_view(request):
    """
    Custom logout view that:
    1. Clears any active chat sessions
    2. Performs Django logout
    3. Redirects to main page
    """
    from django.contrib.auth import logout
    from django.shortcuts import redirect
    from django.contrib import messages
    
    # Clear chat session if exists
    user_id = str(request.user.id)
    if user_id in conversation_memory.conversations:
        conversation_memory.conversations[user_id]['messages'] = []
    
    # Clear any emotion detection state
    try:
        stop_detection()  # Stop emotion detection for this user
    except Exception as e:
        logger.error(f"Error stopping emotion detection during logout: {str(e)}")
    
    # Clear session data
    request.session.flush()
    
    # Perform Django logout
    logout(request)
    
    # Add logout success message
    messages.success(request, 'You have been successfully logged out.')
    
    # Redirect to main page
    return redirect('myapp:main')

@login_required
def profile_page_view(request):
    """Profile page view"""
    return render(request, 'profilepage.html')

@login_required
def edit_profile(request):
    """Edit profile view"""
    return render(request, 'editprofile.html')

@login_required
def dashboard_view(request):
    """Dashboard view"""
    return render(request, 'home.html', {'username': request.user.username})

@login_required
def journal_view(request):
    """Journal view"""
    return render(request, 'journal.html')

@login_required
def change_password_view(request):
    """Change password view"""
    return render(request, 'changepass.html')

@login_required
def new_chat(request):
    """Create a new chat session"""
    # Clear conversation memory for the user
    user_id = str(request.user.id)
    if user_id in conversation_memory.conversations:
        conversation_memory.conversations[user_id]['messages'] = []
    return JsonResponse({
        'status': 'success',
        'message': 'New chat session created',
        'timestamp': datetime.now().isoformat()
    })

@login_required
def chatbot_stats_api(request):
    """Get chatbot interaction statistics"""
    user_id = str(request.user.id)
    context = conversation_memory.get_context(user_id)
    
    if not context:
        return JsonResponse({
            'total_messages': 0,
            'topics_discussed': [],
            'emotion_summary': [],
            'conversation_duration': 0
        })
    
    return JsonResponse({
        'total_messages': context['total_exchanges'],
        'topics_discussed': context['current_topics'],
        'emotion_summary': context['recent_emotions'],
        'conversation_duration': round(context['conversation_duration'], 2)
    })