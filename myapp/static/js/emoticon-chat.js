// Enhanced JavaScript for Emoticon Chatbot Integration

class EmoticonChatInterface {
    constructor() {
        this.chatHistory = [];
        this.currentMood = 'neutral';
        this.isCameraOn = false;
        this.detectionActive = false;
        this.latestDetectedMood = 'neutral';
        this.useInternetSearch = true;
        this.conversationId = null;
        this.isTyping = false;
        
        // DOM elements
        this.chatMessages = document.getElementById('chatbot-messages');
        this.messageInput = document.getElementById('chatbot-input');
        this.sendBtn = document.querySelector('#chatbot-form button[type="submit"]');
        this.typingIndicator = document.getElementById('chatbot-typing');
        this.emotionDetector = document.getElementById('emotionDetector');
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.setupChatInterface();
        this.displayWelcomeMessage();
    }
    
    setupEventListeners() {
        // Chat form submission
        const chatbotForm = document.getElementById('chatbot-form');
        chatbotForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.sendMessage();
        });
        
        // Minimize chat
        const minimizeBtn = document.getElementById('minimize-chat');
        const chatInterface = document.getElementById('chatbot-interface');
        minimizeBtn.addEventListener('click', () => {
            chatInterface.classList.toggle('minimized');
            minimizeBtn.innerHTML = chatInterface.classList.contains('minimized') 
                ? '<i class="fas fa-expand"></i>' 
                : '<i class="fas fa-minus"></i>';
        });

        // New chat
        const newChatBtn = document.getElementById('new-chat');
        newChatBtn.addEventListener('click', () => this.startNewChat());
    }
    
    setupChatInterface() {
        this.scrollToBottom();
    }
    
    displayWelcomeMessage() {
        setTimeout(() => {
            this.addMessage("Hi! I'm Emoticon, your AI emotional companion. How can I help you today?", 'ai');
        }, 1000);
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isTyping) return;
        
        // Add user message immediately
        this.addMessage(message, 'user');
        this.messageInput.value = '';
        this.isTyping = true;
        
        // Show typing indicator
        this.showTypingIndicator(true);
        
        // Prepare request data
        const requestData = {
            message: message,
            emotion: this.emotionDetector ? this.emotionDetector.getAttribute('data-emotion') || '🤖' : '🤖',
            conversation_id: this.conversationId
        };
        
        try {
            const response = await fetch('/chat/api/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCsrfToken()
                },
                body: JSON.stringify(requestData)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Hide typing indicator
            this.showTypingIndicator(false);
            this.isTyping = false;
            
            if (data.response) {
                this.addMessage(data.response, 'ai');
                if (data.conversation_id) {
                    this.conversationId = data.conversation_id;
                }
            }
            
        } catch (error) {
            console.error('Chat error:', error);
            this.showTypingIndicator(false);
            this.isTyping = false;
            this.addMessage("I'm having trouble connecting right now. Please try again in a moment.", 'ai');
        }
    }
    
    addMessage(text, sender) {
        const msgDiv = document.createElement('div');
        msgDiv.style.margin = '8px 0';
        msgDiv.style.textAlign = sender === 'user' ? 'right' : 'left';
        msgDiv.innerHTML = `<span style="display:inline-block;padding:8px 14px;border-radius:16px;max-width:80%;background:${sender==='user'?'#ff4081':'#64b5f6'};color:#fff;font-size:1rem;word-break:break-word;">${text}</span>`;
        this.chatMessages.appendChild(msgDiv);
        this.scrollToBottom();
    }
    
    showTypingIndicator(show) {
        this.typingIndicator.style.display = show ? 'block' : 'none';
        if (show) this.scrollToBottom();
    }
    
    scrollToBottom() {
        setTimeout(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }, 100);
    }
    
    async startNewChat() {
        this.chatMessages.innerHTML = '';
        this.showTypingIndicator(true);
        
        try {
            const response = await fetch('/new_chat/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCsrfToken()
                }
            });
            
            if (!response.ok) throw new Error('Network response was not ok');
            
            this.showTypingIndicator(false);
            this.displayWelcomeMessage();
            this.conversationId = null;
            
        } catch (error) {
            console.error('Error creating new chat:', error);
            this.showTypingIndicator(false);
            this.addMessage("Sorry, I had trouble creating a new chat. Please try again.", 'ai');
        }
    }
    
    getCsrfToken() {
        return document.querySelector('[name=csrfmiddlewaretoken]').value;
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(() => {
        window.emoticonChat = new EmoticonChatInterface();
    }, 1000);
});
