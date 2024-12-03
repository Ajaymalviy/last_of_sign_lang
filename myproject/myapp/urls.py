from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login_views, name='login'), 
    path('signup/', views.signup_view, name='signup'), 
    path('logout/', views.logout, name='logout'), 
    path('speech_to_text/', views.speech_to_text, name='speech_to_text'),
    # path('gif-videocheck/', views.gif_display, name='gif_display'),
    path('tryyy/', views.tryyy, name='tryyy'),
    path('text_to_sign/', views.text_to_sign, name='text_to_sign'),
    path('videocheck/', views.videocheck, name='videocheck'),


]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
