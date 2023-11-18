from django.urls import path
from . import views

urlpatterns = [
    # path('register/', views.register, name='register'),
    # path('login/', views.login_view, name='login'),
    path('login_or_register/', views.login_or_register, name='login_or_register'),
    path('logout/', views.logout_view, name='logout'),
    path('change_password/', views.change_password, name='change_password'),
    # path("verification/", views.verification_sent, name="verification_sent"),
    # path("registercustom/", views.registercustom, name="registercustom"),

]