from django.urls import path, include
from .views import *
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('herbs/', views.herbs, name='herbs'),
    path('illness/', views.illness, name='illness'),
    path('favourite/', views.favourite, name='favourite'),
    path('recognition', views.recognition, name='recognition'),
    path('search/', views.search, name='search'),
    path('add/', views.add, name='add'),
    path('edit/<str:pk>', edit.as_view(), name='edit'),
    path('toggle_favorite/<int:herb_id>/', views.toggle_favorite, name='toggle_favorite'),
]