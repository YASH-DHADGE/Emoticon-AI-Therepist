from django.urls import path
from django.contrib.auth.decorators import login_required
from . import views

app_name = 'myapp'  # Define an application namespace

urlpatterns = [
    # Public pages
    path('', views.main_view, name='main'),
    path('home/', views.home_view, name='home'),
    path('signup/', views.home_view, name='signup'),  # Using home_view as that's what exists
    path('login/', views.login_view, name='login'),  # Custom login view
    
    # Protected pages (require login)
    path('profile/', login_required(views.profile_page_view), name='profile'),
    path('edit-profile/', login_required(views.edit_profile), name='edit_profile'),
    path('dashboard/', login_required(views.dashboard_view), name='dashboard'),
    path('mood_status/', login_required(views.mood_status_view), name='mood_status'),
    path('mood-status/', login_required(views.mood_status_view), name='mood-status'),  # Keep old URL for backward compatibility
    path('journal/', login_required(views.journal_view), name='journal'),
    path('change-password/', login_required(views.change_password_view), name='change_password'),
    
    # Chatbot APIs (all protected)
    path('chat/api/', login_required(views.chatbot_api), name='chatbot_api'),
    path('chat/feedback/', login_required(views.chatbot_feedback_api), name='chatbot_feedback_api'),
    path('chat/update-context/', login_required(views.update_chatbot_context), name='update_chatbot_context'),
    path('new_chat/', login_required(views.new_chat), name='new_chat'),
    path('chat/stats/', login_required(views.chatbot_stats_api), name='chatbot_stats_api'),
    
    # Video streaming (protected)
    path('video_feed/', login_required(views.video_feed), name='video_feed'),
    
    # Authentication URLs
    path('logout/', login_required(views.logout_view), name='logout'),
]
